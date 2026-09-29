"""Execute the frozen diagnostic, ablation, and sequential confirmation protocol."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.decision_evidence import paired_block_interval
from asofcast.experiment import build_examples, run_experiment
from asofcast.preprocessing import partition_bounds, simulate_arrivals, split_origins
from asofcast.research_data import prepare_source
from asofcast.research_diagnosis import (
    confirmatory_gate,
    oracle_acquisition,
    rollout,
    score_rollout,
)
from asofcast.research_learning import (
    LegacyJoint,
    SimplePolicy,
    TorchForecast,
    TreePolicy,
    fit_residual_forecast,
    transition_examples,
)
from asofcast.timeline import Timeline

ROOT = Path(__file__).resolve().parents[1]


def normalized_timeline(bundle, profile=None, arrival_seed=None):
    raw = bundle.timeline
    arrivals = (raw.arrivals if profile is None else
                simulate_arrivals(raw.times, len(raw.columns), arrival_seed, profile=profile))
    return Timeline(raw.times, bundle.scaler.transform(raw.values), arrivals, raw.columns)


def outcomes(bundle, result, y, protocol):
    objective = protocol['objective']
    return score_rollout(result, y, float(bundle.scaler.scale[bundle.manifest['target_channel']]),
                         weight=objective['cost_weight'], delay=objective['delay_weight'],
                         deadline=objective['deadline_seconds'])


def fit_research(bundle, protocol):
    """Read only train/policy/validation origins; freeze choices before evaluation."""
    cfg, raw = bundle.config, bundle.timeline
    target = bundle.manifest['target_channel']
    origins = split_origins(len(raw.times), cfg['lookback'], cfg['horizon'], cfg['stride'])
    bounds = partition_bounds(len(raw.times))
    for part in ('train', 'policy'):
        indices = origins[part]
        cutoff = raw.times[bounds[part][1] - 1]
        origins[part] = indices[raw.arrivals[indices + cfg['horizon'], target] <= cutoff]
    timeline = normalized_timeline(bundle)
    train_x, train_y = build_examples(timeline, origins['train'], cfg, target)
    val_x, val_y = build_examples(timeline, origins['validation'], cfg, target)
    train_flat = train_x.reshape((-1,) + train_x.shape[2:])
    train_labels = np.repeat(train_y, len(cfg['waits_seconds']))
    val_flat = val_x.reshape((-1,) + val_x.shape[2:])
    val_labels = np.repeat(val_y, len(cfg['waits_seconds']))
    base = TorchForecast(bundle.calibrated)
    upgraded, forecast_selection = fit_residual_forecast(
        base, train_flat, train_labels, val_flat, val_labels, protocol['forecast'], seed=cfg['seed'])
    value_only, value_selection = fit_residual_forecast(
        base, train_flat, train_labels, val_flat, val_labels, protocol['forecast'],
        seed=cfg['seed'], metadata=False)
    forecasters = {'base': base, 'upgraded': upgraded, 'value': value_only}
    selection = {'forecast': forecast_selection, 'value_residual': value_selection,
                 'policies': {}, 'split': 'validation', 'origins': len(val_y)}
    probes = {'forecast': upgraded.predict(val_x[:, 0]), 'value': value_only.predict(val_x[:, 0])}
    fitted_policies = {}
    for name, forecaster in forecasters.items():
        features, gains, eligible = transition_examples(
            timeline, origins['policy'], cfg, target, forecaster, seed=cfg['seed'] + 17,
            metadata=name != 'value')
        policy = TreePolicy.fit(features, gains, eligible, cfg['waits_seconds'],
                                 protocol['policy'], seed=cfg['seed'], metadata=name != 'value')
        policy.weight, policy.delay = protocol['objective']['cost_weight'], protocol['objective']['delay_weight']
        candidates = []
        candidate_rollouts = {}
        for margin in protocol['policy']['margins']:
            policy.margin = margin
            result = rollout(timeline, origins['validation'], cfg, forecaster, policy,
                              bundle.acquisition_cost_proxy)
            loss = outcomes(bundle, result, val_y, protocol)
            candidates.append({'margin': margin, 'objective': float(loss['objective'].mean())})
            candidate_rollouts[margin] = result
        chosen = min(candidates, key=lambda r: (r['objective'], -r['margin']))
        policy.margin = chosen['margin']
        fitted_policies[name] = policy
        selection['policies'][name] = {**chosen, 'candidates': candidates}
        probes[f'policy_{name}'] = candidate_rollouts[policy.margin]['actions']
    # A competitive simple comparator can use the same upgraded forecast.
    simple_records = {}
    for forecast_name, forecaster in forecasters.items():
        for wait in range(len(cfg['waits_seconds'])):
            for kind in ('commit', 'target', 'oldest'):
                name = f'{forecast_name}__wait{wait}__{kind}'
                result = rollout(timeline, origins['validation'], cfg, forecaster,
                                  SimplePolicy(wait, kind, target), bundle.acquisition_cost_proxy)
                simple_records[name] = float(outcomes(bundle, result, val_y, protocol)['objective'].mean())
    selection['simple'] = min(simple_records, key=lambda k: (simple_records[k], k))
    selection['simple_candidates'] = simple_records
    return {'forecasters': forecasters, 'policies': fitted_policies, 'selection': selection,
            'selection_probe': probes}


def evaluate_research(bundle, fitted, protocol, profile, arrival_seed):
    cfg, origins = bundle.config, bundle.test_origins
    timeline = normalized_timeline(bundle, profile, arrival_seed)
    target = bundle.manifest['target_channel']
    y = timeline.values[origins + cfg['horizon'], target]
    forecasters, policies = fitted['forecasters'], fitted['policies']
    base, upgraded, value = (forecasters[key] for key in ('base', 'upgraded', 'value'))
    simple_forecast, simple_wait, simple_kind = fitted['selection']['simple'].split('__')
    methods = {
        'legacy_joint': (base, LegacyJoint(bundle)),
        'forecast_only': (upgraded, LegacyJoint(bundle)),
        'policy_only': (base, policies['base']),
        'combined': (upgraded, policies['upgraded']),
        'value_features': (value, policies['value']),
        'validation_simple': (forecasters[simple_forecast], SimplePolicy(int(simple_wait[4:]), simple_kind, target)),
        'base_commit': (base, SimplePolicy()),
        'forecast_commit': (upgraded, SimplePolicy()),
        'value_commit': (value, SimplePolicy()),
        'fixed_1800s': (base, SimplePolicy(1)),
        'fixed_3600s': (base, SimplePolicy(2)),
        'target_sensor': (base, SimplePolicy(0, 'target', target)),
        'oldest_sensor': (base, SimplePolicy(0, 'oldest', target)),
        'random_sensor': (base, SimplePolicy(0, 'random', target, seed=bundle.config['seed'] + arrival_seed)),
        'dlinear_commit': (TorchForecast(bundle.dlinear), SimplePolicy()),
    }
    arrays = {'origins': origins, 'target_standardized': y}
    reports = {}
    for name, (forecaster, policy) in methods.items():
        result = rollout(timeline, origins, cfg, forecaster, policy, bundle.acquisition_cost_proxy)
        loss = outcomes(bundle, result, y, protocol)
        if loss['deadline_violation'].any():
            raise AssertionError('deadline violation')
        reports[name] = {metric: float(values.mean()) for metric, values in loss.items()}
        reports[name]['origins'] = len(origins)
        for metric, values in loss.items():
            arrays[f'{name}__{metric}'] = values
        arrays[f'{name}__actions'] = result['actions']
        arrays[f'{name}__prediction'] = result['prediction']
    return {'methods': reports, 'selection': fitted['selection']}, arrays


def archived_headroom(root=ROOT):
    old = json.loads((root / 'docs/assets/model-evidence/report.json').read_text())
    path = root / 'docs/assets/model-evidence/paired-losses.npz'
    if sha256_file(path) != old['raw_losses_sha256']:
        raise ValueError('archived evidence checksum mismatch')
    rows = {}
    with np.load(path, allow_pickle=False) as arrays:
        for prefix in old['runs']:
            def get(name, prefix=prefix):
                return arrays[f'{prefix}__{name}']
            y, p, cf, eligible = (get(name) for name in ('target_standardized', 'current_standardized',
                                                          'counterfactual_standardized', 'eligible'))
            costs = np.full(eligible.shape[1], np.nan)
            for method in ('mlp', 'ridge', 'oldest', 'target', 'prior'):
                chosen, paid = get(f'acquisition__{method}__choice'), get(f'acquisition__{method}__cost')
                for channel in range(len(costs)):
                    values = paid[chosen == channel]
                    if len(values):
                        if not np.all(values == values[0]) or (np.isfinite(costs[channel]) and costs[channel] != values[0]):
                            raise ValueError('inconsistent archived sensor costs')
                        costs[channel] = values[0]
            if not np.isfinite(costs).all():
                raise ValueError('cannot reconstruct every sensor cost without guessing')
            oracle = oracle_acquisition(y, p, cf, eligible, costs, .03)
            before = np.abs(p.astype(float) - y.astype(float))
            wait_oracle = np.stack([get(f'wait__fixed_{wait}s__objective') for wait in (0, 1800, 3600)]).min(0)
            rows[prefix] = {'commit_objective': float(before.mean()),
                            'oracle_acquisition_objective': float(oracle['objective'].mean()),
                            'oracle_acquisition_gain': float((before - oracle['objective']).mean()),
                            'oracle_wait_gain': float((before - wait_oracle).mean()),
                            'mlp_gain': float((before - get('acquisition__mlp__objective')).mean()),
                            'ridge_gain': float((before - get('acquisition__ridge__objective')).mean()),
                            'oracle_acquisition_rate': float(oracle['acquired'].mean())}
    cells = {}
    for dataset in ('ETTh1', 'ETTh2'):
        for condition in ('mixed_811', 'mixed_1811', 'outage_1811'):
            subset = [row for key, row in rows.items() if key.startswith(dataset + '__') and key.endswith(condition)]
            cell = {metric: float(np.mean([r[metric] for r in subset])) for metric in subset[0]}
            for metric in ('oracle_acquisition_gain', 'oracle_wait_gain', 'mlp_gain', 'ridge_gain'):
                cell[metric + '_relative'] = cell[metric] / cell['commit_objective']
            cells[f'{dataset}/{condition}'] = cell
    return {'source_sha256': sha256_file(path), 'cells': cells, 'runs': rows,
            'scope': 'retrospective oracle with current forecaster; one pull OR waiting separately; not a joint oracle or achievable claim'}


def aggregate(protocol, reports, arrays):
    gate_cfg = protocol['gate']
    cells = {}
    for dataset in protocol['development_datasets'] + protocol['confirmatory_datasets']:
        for condition in protocol['conditions']:
            prefixes = [f'{dataset}__{seed}__{condition["name"]}' for seed in protocol['seeds']]
            if not all(p in reports for p in prefixes):
                continue
            if not all(np.array_equal(arrays[p + '__origins'], arrays[prefixes[0] + '__origins']) for p in prefixes):
                raise ValueError('training seeds must share evaluation origins')
            averages = {method: {metric: float(np.mean([reports[p]['methods'][method][metric] for p in prefixes]))
                                for metric in reports[prefixes[0]]['methods'][method]}
                        for method in reports[prefixes[0]]['methods']}
            comparisons = {}
            for comparator in ('legacy_joint', 'validation_simple'):
                deltas = np.stack([arrays[f'{p}__{comparator}__objective'] - arrays[f'{p}__combined__objective'] for p in prefixes])
                interval = paired_block_interval(deltas, block=gate_cfg['block_origins'],
                    repeats=gate_cfg['bootstrap_repeats'], seed=gate_cfg['bootstrap_seed'])
                interval['sensitivity'] = paired_block_interval(deltas, block=gate_cfg['sensitivity_block_origins'],
                    repeats=gate_cfg['bootstrap_repeats'], seed=gate_cfg['bootstrap_seed'])
                interval['relative_improvement'] = interval['mean'] / max(averages[comparator]['objective'], 1e-12)
                interval['chronological_quarter_improvements'] = [float(part.mean()) for part in np.array_split(deltas.mean(0), 4)]
                comparisons[comparator] = interval
            ablations = {}
            for candidate, comparator in [('forecast_commit', 'base_commit'), ('policy_only', 'legacy_joint'),
                                           ('combined', 'forecast_only'), ('combined', 'value_features')]:
                delta = np.stack([arrays[f'{p}__{comparator}__objective'] - arrays[f'{p}__{candidate}__objective'] for p in prefixes])
                ablations[f'{candidate}_vs_{comparator}'] = paired_block_interval(
                    delta, block=gate_cfg['block_origins'], repeats=gate_cfg['bootstrap_repeats'], seed=gate_cfg['bootstrap_seed'])
            regressions = [max(reports[p]['methods']['combined']['native_error'] /
                               max(reports[p]['methods'][c]['native_error'], 1e-12) - 1
                               for c in ('legacy_joint', 'validation_simple')) for p in prefixes]
            cells[f'{dataset}/{condition["name"]}'] = {'averages': averages, 'comparisons': comparisons, 'ablations': ablations,
                                                        'mae_regressions': regressions}
    expected = [f'{d}/{c["name"]}' for d in protocol['confirmatory_datasets'] for c in protocol['conditions']]
    return cells, confirmatory_gate(cells, expected, min_relative=gate_cfg['minimum_relative_improvement'],
                                     max_regression=gate_cfg['mae_regression_limit'])


def run_suite(protocol_path, data_dir, models_dir, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite results: {out}')
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol['seeds'] != [42, 43, 44]:
        raise ValueError('frozen protocol requires three distinct declared seeds')
    cfg = json.loads((ROOT / protocol['training_config']).read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    source_code = sorted([*ROOT.glob('src/asofcast/*.py'), Path(__file__).resolve()])
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_code}
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'headroom.json', archived_headroom())
    reports, arrays, sources, selections = {}, {}, [], {}
    started = time.monotonic()
    for dataset in protocol['development_datasets'] + protocol['confirmatory_datasets']:
        source, provenance = prepare_source(dataset, data_dir, protocol)
        sources.append(provenance)
        for seed in protocol['seeds']:
            run_key = f'{dataset}-{seed}'
            destination = Path(models_dir) / run_key
            config = {**cfg, 'seed': seed,
                      'target': protocol['preprocessing'].get(dataset, {}).get('target', 'OT')}
            if not destination.exists():
                run_experiment(source, destination, config,
                               source_kind='ett' if dataset == 'ETTh1' else 'user-provided-csv')
            bundle = load_bundle(destination)
            if bundle.config != config or bundle.report['source']['sha256'] != sha256_file(source):
                raise ValueError('cached model does not match protocol and data')
            print(json.dumps({'fitting': run_key, 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
            fitted = fit_research(bundle, protocol)
            selections[run_key] = fitted['selection']
            # Write decisions before the first final-test rollout for this run.
            write_json(out / f'{run_key}-selection.json', fitted['selection'])
            for name, values in fitted['selection_probe'].items():
                arrays[f'{run_key}__validation_probe__{name}'] = values
            for condition in protocol['conditions']:
                prefix = f'{dataset}__{seed}__{condition["name"]}'
                report, values = evaluate_research(bundle, fitted, protocol, condition['profile'], condition['arrival_seed'])
                reports[prefix] = report
                arrays.update({f'{prefix}__{name}': value for name, value in values.items()})
                print(json.dumps({'evaluated': prefix, 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
            write_json(out / f'{run_key}-checkpoint.json', {'reports': {p: reports[p] for p in reports if p.startswith(dataset + '__' + str(seed) + '__')}})
    cells, gate = aggregate(protocol, reports, arrays)
    if hashes != {str(p.relative_to(ROOT)): sha256_file(p) for p in source_code}:
        raise RuntimeError('source code changed during experiment')
    np.savez_compressed(out / 'paired-losses.npz', **arrays)
    report = {'schema': protocol['schema'], 'protocol': protocol,
              'protocol_sha256': sha256_file(protocol_path), 'source_files': hashes,
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                          'sklearn': sklearn.__version__}, 'sources': sources, 'selections': selections,
              'runs': reports, 'cells': cells, 'promotion_gate': gate,
              'elapsed_seconds': time.monotonic() - started, 'raw_losses_sha256': sha256_file(out / 'paired-losses.npz'),
              'limits': ['Arrivals, pull latency (zero), and information costs are simulated.',
                         'The new datasets broaden equipment/geography within energy, not all industrial domains.',
                         'Each dataset is trained separately; not zero-shot transfer.',
                         'Only 4.5 months of Appliances data: final-test block interval has limited power.',
                         'value_features removes arrival metadata from added features, retaining the original calibrated backbone.',
                         'Legacy bundle fitting also generates its old test report automatically; protocol and candidate family were frozen beforehand, and candidate selection never reads that report.',
                         'Oracle headroom is an unachievable hindsight diagnostic, not policy quality.']}
    write_json(out / 'report.json', report)
    print(json.dumps({'complete': str(out), 'promotion_gate': gate}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/research_diagnosis_20260929.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/research-diagnosis')
    parser.add_argument('--models', type=Path, default=ROOT / 'artifacts/research-diagnosis/models')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_suite(args.protocol, args.data, args.models, args.out)
