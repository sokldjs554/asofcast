"""Run the predeclared model evidence protocol without selecting on test outcomes."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from asofcast.acquisition import acquisition_features
from asofcast.bundle import load_bundle
from asofcast.data import git_blob_sha1, sha256_file, write_json
from asofcast.decision_evidence import (
    RidgeGainModel,
    action_losses,
    choose_gain_actions,
    paired_block_interval,
    promotion_gate,
    random_matched_losses,
    select_validation,
)
from asofcast.experiment import (
    _counterfactual_predictions,
    _predict_decisions,
    build_examples,
    build_policy_states,
    run_experiment,
)
from asofcast.policy import choose_steps
from asofcast.preprocessing import partition_bounds, simulate_arrivals, split_origins
from asofcast.timeline import Timeline
from asofcast.training import predict

ROOT = Path(__file__).resolve().parents[1]
SIMPLE_NAMES = ('commit', 'target', 'oldest', 'prior')


def validate_protocol(protocol):
    seeds = protocol['seeds']
    if (len(seeds) < 3 or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('at least three distinct integer training seeds required')
    for field in ('datasets', 'conditions'):
        names = [item['name'] for item in protocol[field]]
        if not names or len(set(names)) != len(names):
            raise ValueError(f'distinct {field} names required')


@dataclass(frozen=True)
class FittedChoices:
    ridge: RidgeGainModel
    prior: np.ndarray
    selection: dict


def feature_cube(x, predictions, costs, remaining):
    return np.stack([acquisition_features(
        x, predictions, np.full(len(x), channel, dtype=np.int64), costs, remaining)
        for channel in range(x.shape[2])], axis=1)


def simple_actions(x, eligible, prior, costs, weight, target):
    n = len(x)
    oldest = np.argmax(np.where(eligible, x[:, -1, :, 2], -np.inf), axis=1)
    return {
        'commit': np.full(n, -1, dtype=np.int64),
        'target': np.where(eligible[:, target], target, -1),
        'oldest': np.where(eligible.any(axis=1), oldest, -1),
        'prior': choose_gain_actions(np.broadcast_to(prior, eligible.shape), eligible, costs, weight),
    }


def fit_choices(bundle, alphas, weight):
    """Only policy/validation origins are accessed here; test outcomes are absent."""
    cfg, raw = bundle.config, bundle.timeline
    target, costs = bundle.manifest['target_channel'], bundle.acquisition_cost_proxy
    origins = split_origins(len(raw.times), cfg['lookback'], cfg['horizon'], cfg['stride'])
    policy_cutoff = raw.times[partition_bounds(len(raw.times))['policy'][1] - 1]
    policy_origins = origins['policy']
    policy_origins = policy_origins[raw.arrivals[policy_origins + cfg['horizon'], target] <= policy_cutoff]
    timeline = Timeline(raw.times, bundle.scaler.transform(raw.values), raw.arrivals, raw.columns)
    x, y = build_examples(timeline, policy_origins, cfg, target)
    current = _predict_decisions(bundle.calibrated, x)
    cf, eligible = _counterfactual_predictions(timeline, policy_origins, cfg, bundle.calibrated, x)
    features, gains, eligibility = [], [], []
    for step, wait in enumerate(cfg['waits_seconds']):
        features.append(feature_cube(x[:, step], current[:, step], costs, 1 - wait / cfg['waits_seconds'][-1]))
        gains.append(np.abs(current[:, step] - y)[:, None] - np.abs(cf[:, step] - y[:, None]))
        eligibility.append(eligible[:, step])
    features, gains, eligible = map(np.concatenate, (features, gains, eligibility))
    prior = np.array([gains[eligible[:, c], c].mean() if eligible[:, c].any() else 0.
                      for c in range(len(costs))])
    ridges = {f'ridge_{alpha:g}': RidgeGainModel.fit(features, gains, eligible, ridge=alpha)
              for alpha in alphas}
    if len(ridges) != len(alphas) or not ridges:
        raise ValueError('nonempty unique ridge alphas required')
    val_x, val_y = build_examples(timeline, origins['validation'], cfg, target)
    val_pred = _predict_decisions(bundle.calibrated, val_x)
    val_cf, val_eligible = _counterfactual_predictions(
        timeline, origins['validation'], cfg, bundle.calibrated, val_x, steps=[0])
    cube = feature_cube(val_x[:, 0], val_pred[:, 0], costs, 1.)
    actions = simple_actions(val_x[:, 0], val_eligible[:, 0], prior, costs, weight, target)
    for name, model in ridges.items():
        actions[name] = choose_gain_actions(model.predict(cube), val_eligible[:, 0], costs, weight)
    loss = {name: action_losses(val_y, val_pred[:, 0], val_cf[:, 0], choice,
                               val_eligible[:, 0], costs, weight, bundle.scaler.scale[target])['objective']
            for name, choice in actions.items()}
    selected_ridge = select_validation({name: loss[name] for name in ridges})
    selected_simple = select_validation({name: loss[name] for name in SIMPLE_NAMES})
    waits = np.asarray(cfg['waits_seconds'])
    fixed_losses = {str(step): np.abs(val_pred[:, step] - val_y) + cfg['delay_cost'] * wait / waits[-1]
                    for step, wait in enumerate(waits)}
    fixed_step = int(select_validation(fixed_losses))
    selection = {
        'split': 'validation', 'ridge': selected_ridge, 'simple': selected_simple,
        'fixed_wait_step': fixed_step, 'validation_origins': len(val_y),
        'candidate_objectives': {name: float(values.mean()) for name, values in loss.items()},
        'fixed_wait_objectives': {name: float(values.mean()) for name, values in fixed_losses.items()},
    }
    return FittedChoices(ridges[selected_ridge], prior, selection)


def evaluate_condition(bundle, fitted, profile, arrival_seed, weight):
    cfg, raw = bundle.config, bundle.timeline
    target, costs = bundle.manifest['target_channel'], bundle.acquisition_cost_proxy
    scale = float(bundle.scaler.scale[target])
    arrivals = simulate_arrivals(raw.times, len(raw.columns), arrival_seed, profile=profile)
    timeline = Timeline(raw.times, bundle.scaler.transform(raw.values), arrivals, raw.columns)
    x, y = build_examples(timeline, bundle.test_origins, cfg, target)
    current = _predict_decisions(bundle.calibrated, x)
    cf, eligible_all = _counterfactual_predictions(
        timeline, bundle.test_origins, cfg, bundle.calibrated, x, steps=[0])
    eligible, cf = eligible_all[:, 0], cf[:, 0]
    cube = feature_cube(x[:, 0], current[:, 0], costs, 1.)
    mlp_gains = predict(bundle.acquisition, cube.reshape(-1, cube.shape[-1])).reshape(eligible.shape)
    actions = simple_actions(x[:, 0], eligible, fitted.prior, costs, weight, target)
    actions['mlp'] = choose_gain_actions(mlp_gains, eligible, costs, weight)
    actions['ridge'] = choose_gain_actions(fitted.ridge.predict(cube), eligible, costs, weight)
    actions['simple'] = actions[fitted.selection['simple']]
    losses = {name: action_losses(y, current[:, 0], cf, choice, eligible, costs, weight, scale)
              for name, choice in actions.items()}
    for name in ('mlp', 'ridge'):
        losses[f'random_{name}'] = random_matched_losses(
            y, current[:, 0], cf, actions[name], eligible, costs, weight, scale)
    arrays = {'origins': bundle.test_origins, 'target_standardized': y,
              'current_standardized': current[:, 0], 'eligible': eligible,
              'counterfactual_standardized': cf}
    acquisition = {}
    for name, loss in losses.items():
        acquisition[name] = {'n': len(y), 'mae': float(loss['native_error'].mean()),
                             'objective': float(loss['objective'].mean()),
                             'mean_cost_proxy': float(loss['cost'].mean()),
                             'acquisition_rate': float(loss['acquired'].mean())}
        for metric, values in loss.items():
            arrays[f'acquisition__{name}__{metric}'] = values
        if name in actions:
            arrays[f'acquisition__{name}__choice'] = actions[name]
    forecast_predictions = {
        'calibrated': current[:, 0], 'persistence': x[:, 0, -1, target, 0],
        'dlinear': predict(bundle.dlinear, x[:, 0]),
        'arrival': predict(bundle.arrival, x[:, 0]),
        'value_only': predict(bundle.value_only, x[:, 0]),
    }
    forecast = {}
    for name, predictions in forecast_predictions.items():
        error = np.abs(predictions.astype(float) - y) * scale
        arrays[f'forecast__{name}__native_error'] = error
        forecast[name] = {'mae': float(error.mean())}
    waits = np.asarray(cfg['waits_seconds'])
    chosen = choose_steps(build_policy_states(x, current, waits.tolist()), bundle.policy,
                          bundle.report['policy_selection']['threshold'], waits)
    wait_choices = {'learned': chosen,
                    'validation_fixed': np.full(len(y), fitted.selection['fixed_wait_step'])}
    wait_choices.update({f'fixed_{int(wait)}s': np.full(len(y), step)
                         for step, wait in enumerate(waits)})
    wait_report = {}
    for name, choices in wait_choices.items():
        error = np.abs(current[np.arange(len(y)), choices] - y)
        objective = error + cfg['delay_cost'] * waits[choices] / waits[-1]
        arrays[f'wait__{name}__objective'] = objective
        arrays[f'wait__{name}__native_error'] = error * scale
        wait_report[name] = {'mae': float(error.mean() * scale), 'objective': float(objective.mean()),
                             'mean_wait_seconds': float(waits[choices].mean())}
    frequencies = np.bincount(chosen, minlength=len(waits)) / len(y)
    random_error = np.abs(current - y[:, None]) @ frequencies
    random_cost = float(frequencies @ waits) / waits[-1] * cfg['delay_cost']
    arrays['wait__random_matched__objective'] = random_error + random_cost
    wait_report['random_matched'] = {'mae': float(random_error.mean() * scale),
                                      'objective': float(random_error.mean() + random_cost),
                                      'mean_wait_seconds': float(frequencies @ waits)}
    return {'selection': fitted.selection, 'acquisition': acquisition,
            'forecast': forecast, 'wait': wait_report}, arrays


def fetch_dataset(dataset, data_dir):
    name = dataset['name']
    if name not in ('ETTh1', 'ETTh2'):
        raise ValueError('unsupported benchmark name')
    path = Path(data_dir) / f'{name}.csv'
    url = f'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv'
    if path.exists():
        body = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=40) as response:
            body = response.read(20_000_001)
    if len(body) > 20_000_000 or git_blob_sha1(body) != dataset['blob_sha1']:
        raise ValueError(f'{name} integrity check failed')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix('.part')
        temporary.write_bytes(body)
        temporary.replace(path)
    return path


def aggregate(protocol, results, raw_arrays):
    cells = {}
    expected = [f"{d['name']}/{c['name']}" for d in protocol['datasets'] for c in protocol['conditions']]
    for key in expected:
        dataset, condition = key.split('/')
        prefixes = [f'{dataset}__{seed}__{condition}' for seed in protocol['seeds']]
        reports = [results[prefix] for prefix in prefixes]
        first_origins = raw_arrays[f'{prefixes[0]}__origins']
        if any(not np.array_equal(first_origins, raw_arrays[f'{p}__origins']) for p in prefixes):
            raise ValueError('paired seed comparison requires identical target origins')
        comparisons = {}
        for baseline in ('mlp', 'simple'):
            delta = np.stack([raw_arrays[f'{p}__acquisition__{baseline}__objective']
                              - raw_arrays[f'{p}__acquisition__ridge__objective'] for p in prefixes])
            estimate = paired_block_interval(delta, block=protocol['block_origins'],
                                             repeats=protocol['bootstrap_repeats'], seed=protocol['bootstrap_seed'])
            estimate['sensitivity'] = paired_block_interval(
                delta, block=protocol['sensitivity_block_origins'],
                repeats=protocol['bootstrap_repeats'], seed=protocol['bootstrap_seed'])
            comparisons[baseline] = estimate
        regressions = [r['acquisition']['ridge']['mae'] / max(r['acquisition']['simple']['mae'], 1e-12) - 1
                       for r in reports]
        averages = {section: {method: {metric: float(np.mean([r[section][method][metric] for r in reports]))
                                       for metric in reports[0][section][method]}
                              for method in reports[0][section]}
                    for section in ('acquisition', 'forecast', 'wait')}
        forecast_delta = np.stack([raw_arrays[f'{p}__forecast__dlinear__native_error']
                                   - raw_arrays[f'{p}__forecast__calibrated__native_error'] for p in prefixes])
        cells[key] = {'comparisons': comparisons, 'mae_relative_regressions': regressions,
                      'averages': averages, 'selections': [r['selection'] for r in reports],
                      'calibrated_vs_dlinear': paired_block_interval(
                          forecast_delta, block=protocol['block_origins'],
                          repeats=protocol['bootstrap_repeats'], seed=protocol['bootstrap_seed'])}
    return cells, promotion_gate(cells, expected, protocol['mae_regression_limit'],
                                 expected_seed_count=len(protocol['seeds']))


def run_protocol(protocol_path, data_dir, out):
    protocol_path, out = Path(protocol_path), Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite evidence: {out}')
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    config_path = ROOT / protocol['training_config']
    config = json.loads(config_path.read_text())
    torch.set_num_threads(config['cpu_threads'])
    out.mkdir(parents=True)
    results, arrays, sources = {}, {}, []
    for dataset in protocol['datasets']:
        source = fetch_dataset(dataset, data_dir)
        sources.append({**dataset, 'sha256': sha256_file(source),
                        'arrival_times': 'simulated', 'measurement_values': 'observed ETT benchmark'})
        for seed in protocol['seeds']:
            cfg = {**config, 'seed': seed}
            destination = out / 'models' / f"{dataset['name']}-{seed}"
            run_experiment(source, destination, cfg,
                           source_kind='ett' if dataset['name'] == 'ETTh1' else 'user-provided-csv')
            bundle = load_bundle(destination)
            fitted = fit_choices(bundle, protocol['ridge_alphas'], protocol['cost_weight'])
            for condition in protocol['conditions']:
                prefix = f"{dataset['name']}__{seed}__{condition['name']}"
                report, raw = evaluate_condition(bundle, fitted, condition['profile'],
                                                  condition['arrival_seed'], protocol['cost_weight'])
                results[prefix] = report
                arrays.update({f'{prefix}__{name}': value for name, value in raw.items()})
                print(json.dumps({'finished': prefix, 'ridge': fitted.selection['ridge'],
                                   'simple': fitted.selection['simple']}, ensure_ascii=False), flush=True)
    cells, gate = aggregate(protocol, results, arrays)
    np.savez_compressed(out / 'paired-losses.npz', **arrays)
    report = {
        'schema': protocol['schema'], 'status': 'experiment_completed',
        'protocol': protocol, 'protocol_sha256': sha256_file(protocol_path),
        'training_config': config, 'training_config_sha256': sha256_file(config_path),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip(),
        'source_file_sha256': {str(path.relative_to(ROOT)): sha256_file(path)
                               for path in sorted([*ROOT.glob('src/asofcast/*.py'), Path(__file__).resolve()])},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__},
        'sources': sources, 'runs': results, 'cells': cells, 'promotion_gate': gate,
        'raw_losses_sha256': sha256_file(out / 'paired-losses.npz'),
        'limits': ['ETTh1 test results were known before this protocol; ETTh2 is additional same-family evidence.',
                   'Each dataset has its own temporal training split; this is not zero-shot transfer.',
                   'Arrival timestamps and information cost are simulated.',
                   'Intervals condition on three training seeds and resample shared temporal blocks.',
                   'One-shot acquisition does not establish superiority of the full joint sequential policy.'],
    }
    write_json(out / 'report.json', report)
    print(json.dumps({'promotion_gate': gate, 'report': str(out / 'report.json')}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/model_evidence_20260928.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/model-evidence')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_protocol(args.protocol, args.data, args.out)
