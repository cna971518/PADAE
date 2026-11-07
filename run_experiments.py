"""Create auditable experiment plans; execute only with --execute."""
from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
import os
from pathlib import Path
import subprocess
import sys
from configuration import BASELINES, LABELS, MODULES, SCENARIOS, SETTINGS, RunConfig
from utils import digest, read_json, sha256_file, write_json

PRESETS = ('comparison', 'ablation', 'clean-ablation', 'mqv-sensitivity', 'cma-grid', 'calibration')


def make_plan(preset, datasets=None, settings=None, scenarios=None, methods=None, seeds=None,
              data_root='data/processed', output_root='results', overrides=None, calibration_file=None):
    if preset not in PRESETS:
        raise ValueError('Unknown experiment preset.')
    datasets = list(datasets or LABELS)
    default_setting = list(SETTINGS) if preset in ('comparison','mqv-sensitivity') else (
        ['4-out-of-10'] if preset == 'ablation' else
        ['1-out-of-10', '4-out-of-20'] if preset == 'calibration' else ['4-out-of-20'])
    settings = list(settings or default_setting)
    scenarios = list(scenarios or (SCENARIOS if preset == 'comparison' else
                     ['clean'] if preset in ('clean-ablation','cma-grid','calibration') else SCENARIOS[1:]))
    methods = list(methods or (list(BASELINES)+['padae'] if preset == 'comparison' else
                   list(MODULES) if preset in ('ablation','clean-ablation') else
                   ['cma_only'] if preset == 'cma-grid' else
                   ['mpdd_only'] if preset == 'calibration' else ['padae']))
    seeds = list(seeds if seeds is not None else (range(42,47) if preset == 'calibration' else
                 [43] if preset == 'mqv-sensitivity' else range(100,105)))
    if preset in ('clean-ablation','cma-grid','calibration') and scenarios != ['clean']:
        raise ValueError('This preset requires only the clean scenario.')
    if preset == 'calibration' and methods != ['mpdd_only']:
        raise ValueError('Calibration requires mpdd_only monitoring.')
    if preset == 'calibration' and calibration_file:
        raise ValueError('Clean monitoring does not load an existing calibration.')
    plans=[]; overrides=overrides or {}
    reserved={'dataset','client_setting','scenario','method','seed','output','data_root','tag',
              'monitor_only','calibration_file','calibration_sha256'}
    if reserved & set(overrides):
        raise ValueError('Identity fields must be set through their dedicated plan arguments, not overrides.')
    for dataset, setting, scenario, method, seed in itertools.product(datasets, settings, scenarios, methods, seeds):
        parameters = [{'tau_mqv': round(i/10,1)} for i in range(3,10)] if preset == 'mqv-sensitivity' else (
            [{'beta':round(b/10,1),'lam':round(l/10,1)} for b in range(1,6) for l in range(5,10)]
            if preset == 'cma-grid' else [{}])
        for par in parameters:
            cfg=RunConfig(dataset=dataset, client_setting=setting, scenario=scenario, method=method,
                          seed=int(seed), data_root=str(data_root), tag=preset,
                          monitor_only=preset == 'calibration',
                          calibration_file=str(calibration_file) if calibration_file else None, **par)
            if set(overrides) & set(par):
                raise ValueError('An override conflicts with a preset parameter sweep.')
            cfg=replace(cfg, **overrides).resolved()
            suffix=''.join(f'_{k}{v:g}' for k,v in par.items())
            run_id=f'{dataset}_{setting}_{scenario}_{method}_s{seed}{suffix}'
            cfg=replace(cfg, output=str(Path(output_root)/preset/run_id))
            plans.append({'run_id':run_id, 'config':cfg.as_dict()})
    ids=[p['run_id'] for p in plans]
    if len(ids)!=len(set(ids)):
        raise ValueError('The experiment plan contains duplicate runs.')
    return plans


def verified_done(directory, config):
    directory=Path(directory)
    if not (directory/'DONE.json').is_file():
        return False
    if read_json(directory/'config.json') != config:
        raise ValueError(f'Configuration mismatch in completed run: {directory}')
    done=read_json(directory/'DONE.json')
    if not done.get('output_hashes'):
        raise ValueError(f'Completion manifest is empty: {directory}')
    for name, expected in done['output_hashes'].items():
        path=directory/name
        if Path(name).name != name or not path.is_file() or sha256_file(path)!=expected:
            raise ValueError(f'Output integrity check failed: {path}')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=PRESETS, required=True)
    p.add_argument('--datasets', nargs='+', choices=tuple(LABELS))
    p.add_argument('--settings', nargs='+', choices=tuple(SETTINGS))
    p.add_argument('--scenarios', nargs='+', choices=SCENARIOS)
    p.add_argument('--methods', nargs='+', choices=tuple(MODULES)+BASELINES)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--data-root', default='data/processed')
    p.add_argument('--output-root', default='results')
    p.add_argument('--plan', default=None)
    p.add_argument('--overrides', help='JSON file containing explicit RunConfig overrides.')
    p.add_argument('--calibration-file', help='Dataset-level MPDD thresholds from clean calibration.')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true', help='Skip only completed, hash-verified runs.')
    a=p.parse_args(argv)
    overrides=read_json(a.overrides) if a.overrides else {}
    plan=make_plan(a.preset,a.datasets,a.settings,a.scenarios,a.methods,a.seeds,
                   a.data_root,a.output_root,overrides,a.calibration_file)
    target=Path(a.plan or str(Path(a.output_root)/f'plan_{a.preset}.json'))
    payload={'preset':a.preset,'planned_runs':len(plan),'runs':plan}
    if target.exists() and read_json(target)!=payload:
        raise FileExistsError('Existing plan differs; select another --plan path.')
    write_json(target,payload)
    print(f'{len(plan)} planned runs; plan: {target}; no training unless --execute is supplied.')
    if not a.execute:
        return 0
    config_dir=target.parent/(target.stem+'_configs'); config_dir.mkdir(parents=True,exist_ok=True)
    failures=[]
    for index,item in enumerate(plan,1):
        cfg=item['config']; directory=Path(cfg['output'])
        if a.resume and verified_done(directory,cfg):
            print(f'{index}/{len(plan)} verified completed: {item["run_id"]}'); continue
        if directory.exists() and any(directory.iterdir()):
            failures.append({'run_id':item['run_id'],'reason':'nonempty_output_not_resumed'})
            continue
        config_path=config_dir/(item['run_id']+'.json'); write_json(config_path,cfg)
        env=dict(os.environ,PYTHONHASHSEED=str(cfg['seed']))
        command=[sys.executable,'-B',str(Path(__file__).with_name('main.py')),'--config',str(config_path)]
        print(f'{index}/{len(plan)} {item["run_id"]}',flush=True)
        code=subprocess.run(command,env=env,check=False).returncode
        if code != 0:
            failures.append({'run_id':item['run_id'],'returncode':code})
    write_json(target.with_name(target.stem+'_execution.json'),
               {'planned_runs':len(plan),'failures':failures,'plan_fingerprint':digest(payload)})
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
