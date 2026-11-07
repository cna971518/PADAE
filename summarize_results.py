"""Aggregate complete fixed-seed groups without mixing data or environments."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from run_experiments import verified_done
from utils import digest, read_json, write_csv, write_json


def protocol_fingerprint(config, result):
    excluded={'seed','output','data_root','expected_data_manifest'}
    return digest({'config':{k:v for k,v in config.items() if k not in excluded},
                   'data':result['data_fingerprint'], 'source':result['source_fingerprint']})


def summarize(roots, expected_seeds, output, split_environments=False):
    expected=set(expected_seeds)
    if len(expected)!=len(expected_seeds) or len(expected)<2:
        raise ValueError('Provide at least two distinct expected seeds.')
    groups=defaultdict(list); discovered=set(); rejected=[]; records=[]
    for root in roots:
        for path in sorted(Path(root).rglob('result.json')):
            path=path.resolve()
            if path in discovered:
                continue
            discovered.add(path)
            config=read_json(path.parent/'config.json')
            if not verified_done(path.parent,config):
                rejected.append({'run':str(path.parent),'reason':'missing_completion_manifest'});continue
            result=read_json(path)
            if result.get('monitor_only'):
                continue
            key=(result['dataset'],result['client_setting'],result['scenario'],result['method'],
                 protocol_fingerprint(config,result))
            if split_environments:
                key += (result['environment_fingerprint'],)
            groups[key].append((result,path.parent,config));records.append(result)
    if not groups:
        raise ValueError('No completed training runs were found.')
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Summary output directory must be empty.')
    output.mkdir(parents=True,exist_ok=True)
    global_rows=[]; class_rows=[]; missing=[]; pairing=defaultdict(set)
    for key,entries in groups.items():
        dataset,setting,scenario,method,protocol=key[:5]
        seeds=[r['seed'] for r,_,_ in entries]
        if len(seeds)!=len(set(seeds)):
            raise ValueError(f'Duplicate training seeds in {dataset}/{setting}/{scenario}/{method}.')
        environments={r['environment_fingerprint'] for r,_,_ in entries}
        if len(environments)>1:
            raise ValueError(f'Mixed environments in {dataset}/{setting}/{scenario}/{method}; '
                             'do not silently average them. Inspect provenance or use --split-environments.')
        if set(seeds)!=expected:
            missing.append({'dataset':dataset,'setting':setting,'scenario':scenario,'method':method,
                            'present_seeds':sorted(seeds),'expected_seeds':sorted(expected),
                            'protocol_fingerprint':protocol})
            continue
        base={'dataset':dataset,'client_setting':setting,'scenario':scenario,'method':method,
              'n':len(entries),'seeds':sorted(seeds),'protocol_fingerprint':protocol,
              'environment_fingerprint':next(iter(environments)),
              'boundary_runs':sum(r['status']!='completed' for r,_,_ in entries)}
        for metric in ('accuracy','macro_f1','validation_accuracy','mqv_ever_tpr','mqv_ever_frr',
                       'mqv_removed_malicious','mpdd_removed_malicious','removed_benign'):
            values=[r.get(metric) for r,_,_ in entries]
            if all(v is None for v in values):
                mean=sd=None
            elif any(v is None for v in values):
                raise ValueError(f'Inconsistent missing values for {metric}.')
            else:
                values=np.asarray(values,dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite result; do not impute or drop seeds.')
                mean=float(values.mean()); sd=float(values.std(ddof=1))
            global_rows.append({**base,'metric':metric,'mean':mean,'sample_sd':sd})
        per_class=defaultdict(list)
        for r,path,cfg in entries:
            if not r.get('initial_weights_sha256'):
                raise ValueError('Missing paired initialization evidence.')
            pair_key=(dataset,setting,scenario,r['seed'],cfg['tag'],cfg['rounds'],cfg['local_epochs'],
                      cfg['batch_size'],cfg['learning_rate'],cfg['tau_mqv'],cfg['tau_r'],cfg['beta'],cfg['lam'])
            pairing[pair_key].add((r['initial_weights_sha256'],r['data_fingerprint'],r['environment_fingerprint']))
            frame=pd.read_csv(path/'per_class_metrics.csv')
            if frame['class'].duplicated().any():
                raise ValueError('Duplicate class metrics.')
            for row in frame.to_dict('records'):
                per_class[row['class']].append(row)
        for name,values in per_class.items():
            if len(values)!=len(entries) or len({v['support'] for v in values})!=1:
                raise ValueError('Class coverage or test support differs between seeds.')
            for metric in ('precision','recall','f1'):
                a=np.asarray([v[metric] for v in values],dtype=float)
                class_rows.append({**base,'class':name,'support':values[0]['support'],'metric':metric,
                                   'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1))})
    incompatible=[{'condition':list(k),'variants':len(v)} for k,v in pairing.items() if len(v)>1]
    write_csv(output/'global_summary.csv',global_rows)
    write_csv(output/'per_class_summary.csv',class_rows)
    write_csv(output/'run_results.csv',records)
    report={'expected_seeds':sorted(expected),'complete_groups':len(global_rows)//8,
            'incomplete_groups':missing,'rejected_runs':rejected,'pairing_conflicts':incompatible,
            'statistical_significance_tests_performed':False}
    write_json(output/'summary_audit.json',report)
    if incompatible:
        raise ValueError('Paired initialization/data/environment mismatch; see summary_audit.json.')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',nargs='+',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(100,105)))
    p.add_argument('--output',required=True)
    p.add_argument('--split-environments',action='store_true')
    a=p.parse_args(argv)
    report=summarize(a.results,a.seeds,a.output,a.split_environments)
    print(f"complete_groups={report['complete_groups']} incomplete_groups={len(report['incomplete_groups'])}")
    return 1 if report['incomplete_groups'] or report['rejected_runs'] else 0


if __name__=='__main__':
    raise SystemExit(main())
