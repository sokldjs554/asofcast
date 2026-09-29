"""Frozen second attempt: conservative finite-depth acquisition policies."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.decision_evidence import paired_block_interval
from asofcast.experiment import RunConfig, run_experiment
from asofcast.information_data import prepare_information_source
from asofcast.information_policy import (
    InformationPolicy,
    fit_backups,
    select_candidate,
    training_transitions,
)
from asofcast.preprocessing import partition_bounds, split_origins
from asofcast.research_diagnosis import confirmatory_gate, rollout
from asofcast.research_learning import LegacyJoint, SimplePolicy, TorchForecast
from scripts.evaluate_research_diagnosis import fit_research, normalized_timeline, outcomes

ROOT = Path(__file__).resolve().parents[1]


def simple_pair(bundle, old):
    name, wait, kind = old['selection']['simple'].split('__')
    return old['forecasters'][name], SimplePolicy(int(wait[4:]), kind, bundle.manifest['target_channel'])


def fit_information(bundle, protocol):
    cfg, target = bundle.config, bundle.manifest['target_channel']
    old = fit_research(bundle, protocol)
    forecaster, simple = simple_pair(bundle, old)
    settings = protocol['information_policy']
    origins = split_origins(len(bundle.timeline.times), cfg['lookback'], cfg['horizon'], cfg['stride'])
    cutoff = bundle.timeline.times[partition_bounds(len(bundle.timeline.times))['policy'][1] - 1]
    parts = []
    for condition in settings['training_conditions']:
        timeline = normalized_timeline(bundle, condition['profile'], condition['arrival_seed'])
        candidates = origins['policy']
        known = timeline.arrivals[candidates + cfg['horizon'], target] <= cutoff
        parts.extend(training_transitions(timeline, candidates[known], cfg, target, forecaster,
            bundle.acquisition_cost_proxy, seed=cfg['seed'] + condition['arrival_seed'],
            weight=protocol['objective']['cost_weight'], delay=protocol['objective']['delay_weight']))
    ensembles = fit_backups(parts, cfg['waits_seconds'], settings, seed=cfg['seed'])
    timeline = normalized_timeline(bundle)
    val_origins = origins['validation']
    y = timeline.values[val_origins + cfg['horizon'], target]
    simple_result = rollout(timeline, val_origins, cfg, forecaster, simple, bundle.acquisition_cost_proxy)
    simple_scores = outcomes(bundle, simple_result, y, protocol)
    records, probes, policies, choices = {}, {}, {}, {}
    for name, depth, uncertainty in [('information', settings['primary_depth'], settings['primary_uncertainty']),
                                      ('myopic', 1, 1.), ('decision_penalty_removed', 2, 0.)]:
        candidates, available = {'simple': simple_scores}, {'simple': simple}
        actions = {'simple': simple_result['actions']}
        for margin in settings['margins']:
            key = f'margin_{margin}'
            policy = InformationPolicy(ensembles[depth], cfg['waits_seconds'], uncertainty=uncertainty,
                margin=margin, weight=protocol['objective']['cost_weight'], delay=protocol['objective']['delay_weight'])
            result = rollout(timeline, val_origins, cfg, forecaster, policy, bundle.acquisition_cost_proxy)
            candidates[key] = outcomes(bundle, result, y, protocol)
            available[key], actions[key] = policy, result['actions']
            probes[f'{name}_{key}_actions'] = result['actions']
            probes[f'{name}_{key}_prediction'] = result['prediction']
        choice = select_candidate(candidates)
        policies[name], choices[name] = available[choice], choice
        records[name] = {k: {metric: float(a.mean()) for metric, a in values.items()}
                         for k, values in candidates.items()}
        probes[f'{name}_selected_actions'] = actions[choice]
    selection = dict(forecast=old['selection']['simple'].split('__')[0],
                     strong_simple=old['selection']['simple'], choices=choices,
                     candidates=records['information'], ablation_candidates=records,
                     selected=records['information'][choices['information']],
                     old_selection=old['selection'], validation_origins=len(val_origins),
                     training_rows=sum(len(p['features']) for p in parts), split='validation')
    return dict(old=old, forecaster=forecaster, policies=policies, selection=selection, probes=probes)


def evaluate_information(bundle, fitted, protocol, condition):
    timeline = normalized_timeline(bundle, condition['profile'], condition['arrival_seed'])
    cfg, target, origins = bundle.config, bundle.manifest['target_channel'], bundle.test_origins
    y = timeline.values[origins + cfg['horizon'], target]
    old, forecaster = fitted['old'], fitted['forecaster']
    simple_forecaster, simple = simple_pair(bundle, old)
    methods = {name: (forecaster, policy) for name, policy in fitted['policies'].items()}
    methods.update(legacy_joint=(old['forecasters']['base'], LegacyJoint(bundle)),
                   prior_combined=(old['forecasters']['upgraded'], old['policies']['upgraded']),
                   validation_simple=(simple_forecaster, simple),
                   same_forecast_commit=(forecaster, SimplePolicy()),
                   same_forecast_target=(forecaster, SimplePolicy(0, 'target', target)),
                   same_forecast_oldest=(forecaster, SimplePolicy(0, 'oldest', target)),
                   same_forecast_wait_half=(forecaster, SimplePolicy(1)),
                   same_forecast_wait_deadline=(forecaster, SimplePolicy(2)),
                   dlinear_commit=(TorchForecast(bundle.dlinear), SimplePolicy()))
    reports, arrays = {}, {'origins': origins, 'target_standardized': y}
    for name, (forecast, policy) in methods.items():
        result = rollout(timeline, origins, cfg, forecast, policy, bundle.acquisition_cost_proxy)
        scores = outcomes(bundle, result, y, protocol)
        if scores['deadline_violation'].any():
            raise AssertionError('deadline violation')
        reports[name] = {metric: float(values.mean()) for metric, values in scores.items()}
        for key in ('actions', 'prediction', 'target_times'):
            arrays[f'{name}__{key}'] = result[key]
        arrays.update({f'{name}__{key}': value for key, value in scores.items()})
    return dict(methods=reports, selection=fitted['selection']), arrays


def aggregate(protocol, reports, arrays):
    cells = {}
    gate = protocol['gate']
    for dataset in protocol['confirmatory_datasets']:
        for condition in protocol['conditions']:
            prefixes = [f'{dataset}__{seed}__{condition["name"]}' for seed in protocol['seeds']]
            if not all(np.array_equal(arrays[f'{prefixes[0]}__origins'], arrays[f'{p}__origins']) for p in prefixes):
                raise ValueError('paired origins differ across training seeds')
            averages = {name: {metric: float(np.mean([reports[p]['methods'][name][metric] for p in prefixes]))
                              for metric in reports[prefixes[0]]['methods'][name]}
                        for name in reports[prefixes[0]]['methods']}
            comparisons = {}
            for comparator in ('legacy_joint', 'validation_simple', 'prior_combined', 'myopic', 'decision_penalty_removed'):
                delta = np.stack([arrays[f'{p}__{comparator}__objective'] - arrays[f'{p}__information__objective']
                                  for p in prefixes])
                interval = paired_block_interval(delta, block=gate['block_origins'],
                    repeats=gate['bootstrap_repeats'], seed=gate['bootstrap_seed'])
                interval['relative_improvement'] = interval['mean'] / max(averages[comparator]['objective'], 1e-12)
                interval['sensitivity'] = paired_block_interval(delta, block=gate['sensitivity_block_origins'],
                    repeats=gate['bootstrap_repeats'], seed=gate['bootstrap_seed'])
                interval['chronological_quarter_improvements'] = [float(a.mean()) for a in np.array_split(delta.mean(0), 4)]
                comparisons[comparator] = interval
            regression = [max(reports[p]['methods']['information']['native_error'] /
                              max(reports[p]['methods'][c]['native_error'], 1e-12) - 1
                              for c in ('legacy_joint', 'validation_simple')) for p in prefixes]
            cells[f'{dataset}/{condition["name"]}'] = dict(averages=averages, comparisons=comparisons,
                                                          mae_regressions=regression)
    expected = [f'{d}/{c["name"]}' for d in protocol['confirmatory_datasets'] for c in protocol['conditions']]
    return cells, confirmatory_gate(cells, expected, min_relative=gate['minimum_relative_improvement'],
                                   max_regression=gate['mae_regression_limit'])


def effective_protocol(protocol, dataset):
    result = {**protocol, 'objective': dict(protocol['objective'])}
    result['objective']['deadline_seconds'] = protocol.get('new_data', {}).get(dataset, {}).get('grid_seconds', 3600)
    return result


def run(protocol_path, out, models, data, *, development=False):
    protocol_path, out, models, data = map(Path, (protocol_path, out, models, data))
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    protocol = json.loads(protocol_path.read_text())
    protocol_digest = sha256_file(protocol_path)
    if protocol['seeds'] != [42, 43, 44]:
        raise ValueError('three distinct frozen training seeds required')
    cfg = json.loads((ROOT / protocol['training_config']).read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    source_paths = sorted([*ROOT.glob('src/asofcast/*.py'), Path(__file__), ROOT / 'scripts/evaluate_research_diagnosis.py'])
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths}
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', protocol)
    reports, arrays, selections, sources, bundles = {}, {}, {}, {}, {}
    started = time.monotonic()
    datasets = protocol['development_datasets'] if development else protocol['confirmatory_datasets']
    seeds = [42] if development else protocol['seeds']
    for dataset in datasets:
        effective = effective_protocol(protocol, dataset)
        if not development:
            source, provenance = prepare_information_source(dataset, data, protocol)
            sources[dataset] = provenance
        for seed in seeds:
            key = f'{dataset}-{seed}'
            destination = models / key
            if not development:
                deadline = effective['objective']['deadline_seconds']
                config = RunConfig.model_validate({**cfg, 'seed': seed,
                    'waits_seconds': [0, deadline / 2, deadline],
                    'target': protocol['new_data'][dataset]['target']}).model_dump()
                if not destination.exists():
                    print(json.dumps({'training_base': key}), flush=True)
                    run_experiment(source, destination, config, source_kind='user-provided-csv')
            bundle = load_bundle(destination)
            if not development and (bundle.config != config or bundle.report['source']['sha256'] != sha256_file(source)):
                raise ValueError('bundle configuration/source mismatch')
            bundles[key] = dict(manifest=bundle.manifest, config=bundle.config,
                                manifest_sha256=sha256_file(destination / 'manifest.json'))
            print(json.dumps({'fitting_policy': key, 'elapsed': round(time.monotonic() - started, 1)}), flush=True)
            fitted = fit_information(bundle, effective)
            selections[key] = fitted['selection']
            write_json(out / f'{key}-selection.json', fitted['selection'])
            arrays.update({f'{key}__validation__{name}': value for name, value in fitted['probes'].items()})
            print(json.dumps({'selected': key, 'choices': fitted['selection']['choices']}), flush=True)
            if not development:
                for condition in protocol['conditions']:
                    prefix = f'{dataset}__{seed}__{condition["name"]}'
                    report, values = evaluate_information(bundle, fitted, effective, condition)
                    reports[prefix] = report
                    arrays.update({f'{prefix}__{name}': value for name, value in values.items()})
                    print(json.dumps({'evaluated': prefix, 'elapsed': round(time.monotonic() - started, 1)}), flush=True)
            write_json(out / 'progress.json', dict(selections=selections, reports=reports))
    cells, gate = ({}, {'passed': False, 'reason': 'development only; no confirmatory test evaluated'}) if development else aggregate(protocol, reports, arrays)
    if (hashes != {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths}
            or sha256_file(protocol_path) != protocol_digest):
        raise RuntimeError('source code or protocol changed during execution')
    np.savez_compressed(out / 'paired-losses.npz', **arrays)
    report = dict(schema=protocol['schema'], development_only=development, protocol=protocol,
        protocol_sha256=sha256_file(protocol_path), source_files=hashes,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__, sklearn=sklearn.__version__),
        sources=sources, bundles=bundles, selections=selections, runs=reports, cells=cells, promotion_gate=gate,
        raw_losses_sha256=sha256_file(out / 'paired-losses.npz'), elapsed_seconds=time.monotonic() - started)
    write_json(out / 'report.json', report)
    print(json.dumps({'complete': str(out), 'gate': gate}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/information_value_20260929.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/information-value')
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    run(args.protocol, args.out, args.models, args.data, development=args.development)
