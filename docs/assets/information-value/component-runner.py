import json
import time
from pathlib import Path
import numpy as np
import torch
from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from scripts.evaluate_information_value import effective_protocol
from scripts.evaluate_research_diagnosis import fit_research, evaluate_research, aggregate

root=Path.cwd()
source=root/'docs/assets/information-value/report.json'
primary=json.loads(source.read_text())
protocol=primary['protocol']
# No new choice or model family: call only the implementation frozen before the primary run.
for path,digest in primary['source_files'].items():
 assert sha256_file(root/path)==digest,path
assert sha256_file(root/'configs/information_value_20260929.json')==primary['protocol_sha256']
out=root/'artifacts/information-value/component-supplement'
out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
reports,arrays={},{}
started=time.monotonic()
with np.load(root/'docs/assets/information-value/paired-losses.npz',allow_pickle=False) as original:
 for dataset in protocol['confirmatory_datasets']:
  effective=effective_protocol(protocol,dataset)
  for seed in protocol['seeds']:
   key=f'{dataset}-{seed}'
   bundle=load_bundle(root/'artifacts/information-value/models1'/key)
   fitted=fit_research(bundle,effective)
   assert fitted['selection']==primary['selections'][key]['old_selection']
   print(json.dumps({'fitted':key,'elapsed':round(time.monotonic()-started,1)}),flush=True)
   for condition in protocol['conditions']:
    prefix=f'{dataset}__{seed}__{condition["name"]}'
    report,values=evaluate_research(bundle,fitted,effective,condition['profile'],condition['arrival_seed'])
    # These fixed-name methods denote steps, not absolute seconds on GasCO.
    names={'fixed_1800s':'fixed_half','fixed_3600s':'fixed_deadline'}
    report['methods']={names.get(k,k):v for k,v in report['methods'].items()}
    values={names.get(k.split('__')[0],k.split('__')[0])+('__'+k.split('__',1)[1] if '__' in k else ''):v for k,v in values.items()}
    for old,new in [('legacy_joint','legacy_joint'),('validation_simple','validation_simple'),('combined','prior_combined'),('dlinear_commit','dlinear_commit')]:
     for metric in ['prediction','actions',*primary['runs'][prefix]['methods'][new]]:
      assert np.array_equal(values[f'{old}__{metric}'],original[f'{prefix}__{new}__{metric}']),(prefix,old,metric)
    reports[prefix]=report
    arrays.update({f'{prefix}__{k}':v for k,v in values.items()})
    print(json.dumps({'evaluated':prefix}),flush=True)
cells,prior_gate=aggregate(protocol,reports,arrays)
for path,digest in primary['source_files'].items(): assert sha256_file(root/path)==digest
np.savez_compressed(out/'paired-losses.npz',**arrays)
write_json(out/'report.json',dict(schema='asofcast.information-component-supplement.v1',
 scope='Descriptive completion of the inherited previous-component comparisons after primary test exposure. No tuning or promotion. Primary report and gate remain unchanged.',
 frozen_source_files=primary['source_files'],primary_report_sha256=sha256_file(source),protocol_sha256=primary['protocol_sha256'],
 old_family='forecast_only, policy_only, combined, value_features refer to the earlier residual-forecast/one-stage policy family; not component ablations of the new InformationPolicy',
 value_features_scope='Removes metadata from added residual/one-stage tree features, not from the base calibrated forecaster.',
 step_label_aliases={'fixed_1800s':'fixed_half','fixed_3600s':'fixed_deadline'},
 overlapping_primary_methods_exact=['legacy_joint','validation_simple','combined=prior_combined','dlinear_commit'],
 fits='Refit frozen research models on the existing run1 base bundles; validation selections exactly match primary old_selection.',
 runtime=primary['runtime'],runs=reports,cells=cells,
 previous_combined_gate_descriptive=prior_gate,primary_gate=primary['promotion_gate'],
 raw_losses_sha256=sha256_file(out/'paired-losses.npz'),arrays=len(arrays),elapsed_seconds=time.monotonic()-started))
print(json.dumps({'complete':str(out),'arrays':len(arrays),'primary_gate':primary['promotion_gate']['passed']}),flush=True)
