"""Read-only audit of frozen information-value evidence; no fitting or evaluation.

Usage: .venv/bin/python /tmp/asofcast-information-independent-audit.py REPO_ROOT
Writes only a JSON summary to stdout. Uses independently written objective,
trace-accounting, bootstrap, selection, and gate calculations. The only project
helper imported is the frozen arrival simulator, to verify trace eligibility.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / 'src'))
from asofcast.preprocessing import simulate_arrivals

base = root / 'artifacts/information-value'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
reports = [read(base / f'confirmation{i}/report.json') for i in (1, 2)]
arrays = []
for i, report in enumerate(reports, 1):
    path = base / f'confirmation{i}/paired-losses.npz'
    assert sha(path) == report['raw_losses_sha256']
    with np.load(path, allow_pickle=False) as archive:
        arrays.append({key: archive[key] for key in archive.files})
assert arrays[0].keys() == arrays[1].keys()
for key in arrays[0]:
    np.testing.assert_array_equal(arrays[0][key], arrays[1][key], err_msg=key)
for field in ('schema', 'development_only', 'protocol', 'protocol_sha256',
              'source_files', 'code_commit', 'runtime', 'sources', 'selections',
              'runs', 'cells', 'promotion_gate'):
    assert reports[0][field] == reports[1][field], field
report, data = reports[0], arrays[0]
protocol = report['protocol']
assert report['code_commit'] == 'b8d76a57da6aba3146d81ab3c2917ba38256b187'
for path, digest in report['source_files'].items():
    assert sha(root / path) == digest, path
assert sha(root / 'configs/information_value_20260929.json') == report['protocol_sha256']

bundle_files = checkpoint_pairs = checkpoint_tensors = metrics = traces = 0
origin_counts = {}
for key, recorded in report['bundles'].items():
    for i, run in enumerate(reports, 1):
        directory = base / f'models{i}' / key
        entry = run['bundles'][key]
        assert sha(directory / 'manifest.json') == entry['manifest_sha256']
        assert read(directory / 'config.json') == entry['config']
        for name, digest in entry['manifest']['files'].items():
            assert sha(directory / name) == digest, (i, key, name)
            bundle_files += 1
    for name in recorded['manifest']['files']:
        if name.endswith('.pt'):
            left = base / 'models1' / key / name
            right = base / 'models2' / key / name
            assert sha(left) == sha(right), (key, name)
            a = torch.load(left, map_location='cpu', weights_only=True)
            b = torch.load(right, map_location='cpu', weights_only=True)
            assert a.keys() == b.keys()
            for tensor in a:
                assert torch.equal(a[tensor], b[tensor]), (key, name, tensor)
                checkpoint_tensors += 1
            checkpoint_pairs += 1
    assert read(base / 'models1' / key / 'scaler.json') == read(base / 'models2' / key / 'scaler.json')

for dataset in protocol['confirmatory_datasets']:
    seed_costs = []
    for seed in protocol['seeds']:
        key = f'{dataset}-{seed}'
        directory = base / 'models1' / key
        entry = report['bundles'][key]
        cfg, manifest = entry['config'], entry['manifest']
        scaler = read(directory / 'scaler.json')
        target = manifest['target_channel']
        scale, mean = scaler['scale'][target], scaler['mean'][target]
        with np.load(directory / 'replay.npz', allow_pickle=False) as replay:
            times, values, origins = replay['times'], replay['values'], replay['test_origins']
        origin_counts[dataset] = len(origins)
        waits = np.asarray(cfg['waits_seconds'])
        costs = np.asarray(manifest['acquisition_cost_proxy'])
        seed_costs.append(costs)
        selection = report['selections'][key]
        choices = selection['old_selection']['simple_candidates']
        simple_name = min(choices, key=lambda name: (choices[name], name))
        assert selection['strong_simple'] == simple_name
        assert selection['forecast'] == simple_name.split('__')[0]
        candidate_scores = selection['candidates']
        simple_score = candidate_scores['simple']
        valid = {name: score for name, score in candidate_scores.items()
                 if score['native_error'] <= simple_score['native_error'] * 1.01}
        winner = min(valid, key=lambda name: (valid[name]['objective'], name != 'simple', name))
        assert winner == selection['choices']['information']
        assert selection['selected']['objective'] <= simple_score['objective']
        for condition in protocol['conditions']:
            prefix = f'{dataset}__{seed}__{condition["name"]}'
            run = report['runs'][prefix]
            np.testing.assert_array_equal(origins, data[prefix + '__origins'])
            native_y = values[origins + cfg['horizon'], target]
            y = (native_y - mean) / scale
            np.testing.assert_array_equal(y, data[prefix + '__target_standardized'])
            arrivals = simulate_arrivals(times, len(costs), condition['arrival_seed'], condition['profile'])
            deadline = protocol['new_data'][dataset]['grid_seconds']
            assert int(times[1] - times[0]) == deadline == waits[-1]
            for method, means in run['methods'].items():
                get = lambda metric: data[f'{prefix}__{method}__{metric}']
                prediction = get('prediction')
                native_error = np.abs(prediction * scale + mean - native_y)
                np.testing.assert_allclose(native_error, get('native_error'), rtol=1e-12, atol=1e-10)
                paid, waited, counts = (np.zeros(len(origins)) for _ in range(3))
                for row, trace in enumerate(get('actions')):
                    step, acquired, committed = 0, set(), False
                    for action in trace:
                        if action == -9:
                            assert committed
                            continue
                        assert not committed
                        if action == -1:
                            committed = True
                        elif action == -2:
                            step += 1
                            assert step < len(waits)
                        else:
                            assert 0 <= action < len(costs) and action not in acquired
                            assert arrivals[origins[row], action] > times[origins[row]] + waits[step]
                            paid[row] += costs[action]
                            acquired.add(action)
                    assert committed
                    waited[row], counts[row] = waits[step], len(acquired)
                traces += len(origins)
                for metric, recomputed in (('cost', paid), ('wait_seconds', waited), ('acquired_count', counts)):
                    np.testing.assert_array_equal(recomputed, get(metric))
                np.testing.assert_array_equal(get('target_times'), times[origins] + cfg['horizon'] * deadline)
                objective = np.abs(prediction - y) + .03 * paid + .02 * waited / deadline
                np.testing.assert_array_equal(objective, get('objective'))
                assert not get('deadline_violation').any()
                for metric, expected in means.items():
                    assert np.isclose(get(metric).mean(), expected, rtol=0, atol=1e-12), (prefix, method, metric)
                    metrics += 1
            if selection['choices']['information'] == 'simple':
                for metric in ('actions', 'prediction', 'objective', 'native_error', 'cost', 'wait_seconds'):
                    np.testing.assert_array_equal(data[f'{prefix}__information__{metric}'],
                                                  data[f'{prefix}__validation_simple__{metric}'])
    for costs in seed_costs[1:]:
        np.testing.assert_array_equal(costs, seed_costs[0])

intervals = 0
gate_checks = {}
for dataset in protocol['confirmatory_datasets']:
    for condition in protocol['conditions']:
        name = condition['name']
        key = f'{dataset}/{name}'
        cell = report['cells'][key]
        prefixes = [f'{dataset}__{seed}__{name}' for seed in protocol['seeds']]
        for prefix in prefixes[1:]:
            np.testing.assert_array_equal(data[prefix + '__origins'], data[prefixes[0] + '__origins'])
        for comparator, interval in cell['comparisons'].items():
            delta = np.stack([data[f'{p}__{comparator}__objective'] - data[f'{p}__information__objective']
                              for p in prefixes])
            series = delta.mean(axis=0)
            for block, record in ((42, interval), (6, interval['sensitivity'])):
                rng = np.random.default_rng(20260929)
                n = len(series)
                starts = rng.integers(0, n, size=(2000, math.ceil(n / block)))
                samples = [series[((row[:, None] + np.arange(block)) % n).reshape(-1)[:n]].mean()
                           for row in starts]
                lower, upper = np.quantile(samples, [.025, .975])
                assert lower == record['lower95'] and upper == record['upper95'], (key, comparator, block)
                intervals += 1
            assert float(series.mean()) == interval['mean']
            np.testing.assert_array_equal(delta.mean(axis=1), interval['seed_improvements'])
            denominator = np.mean([data[f'{p}__{comparator}__objective'].mean() for p in prefixes])
            assert interval['relative_improvement'] == interval['mean'] / denominator
        regressions = [max(data[f'{p}__information__native_error'].mean() /
                           data[f'{p}__{c}__native_error'].mean() - 1
                           for c in ('legacy_joint', 'validation_simple')) for p in prefixes]
        np.testing.assert_array_equal(regressions, cell['mae_regressions'])
        checks = {c: bool(cell['comparisons'][c]['lower95'] > 0
                         and cell['comparisons'][c]['relative_improvement'] >= .01
                         and np.all(np.asarray(cell['comparisons'][c]['seed_improvements']) > 0))
                  for c in ('legacy_joint', 'validation_simple')}
        checks['mae'] = bool(np.all(np.asarray(regressions) <= .01))
        gate_checks[key] = checks
assert all(all(checks.values()) for checks in gate_checks.values()) == report['promotion_gate']['passed'] == False
print(json.dumps(dict(passed=True, arrays_exact=len(data),
    npz_hash_exact=reports[0]['raw_losses_sha256'] == reports[1]['raw_losses_sha256'],
    verified_bundle_files=bundle_files, checkpoint_pairs_exact=checkpoint_pairs,
    checkpoint_tensors_exact=checkpoint_tensors, verified_report_metrics=metrics,
    validated_origin_action_traces=traces, independent_bootstrap_intervals=intervals,
    origins_per_seed_condition=origin_counts, gate_checks=gate_checks,
    promotion_gate_passed=False), indent=2))
