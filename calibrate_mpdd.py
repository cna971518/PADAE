"""Generate clean monitoring runs and calculate dataset-level MPDD thresholds."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import pandas as pd
from configuration import LABELS
from run_experiments import main as run_batch, verified_done
from utils import read_json, write_json

TRAINING_FIELDS = ('rounds', 'local_epochs', 'batch_size', 'learning_rate',
                   'warmup_rounds', 'confirmation_rounds', 'trim_batches')


def truth(value):
    return str(value).strip().lower() in ('true', '1')


def calibration_candidates(rows):
    previous, values, seen = {}, [], set()
    for row in sorted(rows, key=lambda r: int(r['round'])):
        round_id, episode = int(row['round']), int(row['episode'])
        if round_id in seen:
            raise ValueError('Duplicate round in a calibration trajectory.')
        seen.add(round_id)
        ratio = row.get('R')
        structural = truth(row.get('structural_valid', False))
        warmup = truth(row.get('warmup', True))
        if not structural or warmup or ratio is None or not math.isfinite(float(ratio)):
            previous = {}
            continue
        suspicious = row.get('suspicious_ids', [])
        if isinstance(suspicious, str):
            suspicious = json.loads(suspicious)
        if not isinstance(suspicious, list) or len(set(suspicious)) != len(suspicious):
            raise ValueError('Invalid suspicious-client list.')
        current = {}
        for client in suspicious:
            if isinstance(client, bool) or int(client) != client or client < 0:
                raise ValueError('Client IDs must be nonnegative integers.')
            key = (episode, int(client))
            if key in previous and previous[key][0] == round_id - 1:
                values.append({'client_id': int(client), 'episode': episode,
                               'rounds': [round_id - 1, round_id],
                               'critical_R': min(previous[key][1], float(ratio))})
            current[key] = (round_id, float(ratio))
        previous = current
    return values


def calibrate(roots, margin):
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('The calibration margin must be positive and finite.')
    by_dataset, provenance = defaultdict(list), defaultdict(list)
    protocols, training = defaultdict(set), {}
    training_data = defaultdict(dict)
    seen, identities = set(), set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f'Calibration run directory not found: {root}')
        configurations = sorted(root.rglob('config.json'))
        if not configurations:
            raise ValueError(f'No calibration trajectories found in {root}.')
        for path in configurations:
            directory = path.parent.resolve()
            if directory in seen:
                continue
            seen.add(directory)
            cfg = read_json(path)
            if not (cfg.get('scenario') == 'clean' and cfg.get('method') == 'mpdd_only'
                    and cfg.get('monitor_only') is True):
                raise ValueError('Calibration requires clean, MPDD-only monitoring without removals.')
            if cfg['confirmation_rounds'] != 2 or cfg['warmup_rounds'] != 2:
                raise ValueError('This calibration rule requires two warmup and two confirmation rounds.')
            if not verified_done(directory, cfg):
                raise ValueError(f'Incomplete calibration trajectory: {directory.name}')
            result = read_json(directory / 'result.json')
            if (result['status'] != 'completed' or result['completed_rounds'] != cfg['rounds']
                    or result['planned_rounds'] != cfg['rounds']):
                raise ValueError('Calibration trajectories must complete all planned rounds.')
            if result['initial_malicious'] != 0 or result['removed_benign'] != 0:
                raise ValueError('Clean calibration must not contain attacks or client removal.')
            dataset, seed, count = cfg['dataset'], cfg['seed'], result['K']
            if result['dataset'] != dataset or result['seed'] != seed:
                raise ValueError('Calibration result/configuration mismatch.')
            identity = (dataset, count, seed)
            if identity in identities:
                raise ValueError('Duplicate dataset/client-count/seed in calibration.')
            identities.add(identity)
            protocol = {key: cfg[key] for key in TRAINING_FIELDS}
            if dataset in training and training[dataset] != protocol:
                raise ValueError('Do not mix training protocols within a dataset calibration.')
            training[dataset] = protocol
            manifest = read_json(directory / 'data_manifest.json')
            files = {row['name']: row['sha256'] for row in manifest['files']
                     if row['name'] == 'Val.csv' or row['name'].endswith('_train.csv')}
            required = {f'client_{i:02d}_train.csv' for i in range(1, count + 1)} | {'Val.csv'}
            if set(files) != required:
                raise ValueError('Calibration training/validation data coverage is incomplete.')
            prior = training_data[dataset].get(str(count))
            if prior is not None and prior != files:
                raise ValueError('Calibration seeds must use the same fixed data partitions.')
            training_data[dataset][str(count)] = files
            rows = pd.read_csv(directory / 'cluster_round_summary.csv').to_dict('records')
            if len(rows) != cfg['rounds'] or {int(r['round']) for r in rows} != set(range(1, cfg['rounds'] + 1)):
                raise ValueError('Calibration round records are incomplete or duplicated.')
            evidence = calibration_candidates(rows)
            for row in evidence:
                if row['client_id'] >= count:
                    raise ValueError('Calibration client ID exceeds the configured client count.')
                row.update(run=directory.name, seed=seed, K=count)
            by_dataset[dataset].extend(evidence)
            protocols[dataset].add((result['source_fingerprint'], result['environment_fingerprint'],
                                   result.get('model_backend')))
            provenance[dataset].append({'run': directory.name, 'seed': seed, 'K': count,
                                        'data_fingerprint': result['data_fingerprint']})
    output = {}
    for dataset, runs in provenance.items():
        if len(protocols[dataset]) != 1:
            raise ValueError('Calibration source/environment differs; use consistent runs.')
        counts = sorted({r['K'] for r in runs})
        seed_sets = [{r['seed'] for r in runs if r['K'] == count} for count in counts]
        if any(seeds != seed_sets[0] for seeds in seed_sets):
            raise ValueError('Each calibrated client count must contain the same declared seeds.')
        evidence = by_dataset[dataset]
        if not evidence:
            raise ValueError(f'No eligible two-round evidence for {dataset}; extend clean monitoring.')
        critical = max(row['critical_R'] for row in evidence)
        tau = critical + margin
        if not math.isfinite(tau) or tau <= max(1.0, critical):
            raise ValueError('The margin must produce a finite threshold above both 1 and the critical ratio.')
        output[dataset] = {
            'tau_r': tau, 'margin': margin, 'max_two_round_min_R': critical,
            'rule': 'max eligible consecutive two-round minimum plus margin',
            'client_counts': counts, 'seeds': sorted(seed_sets[0]), 'trajectories': len(runs),
            'training_protocol': training[dataset], 'training_data': training_data[dataset],
            'source_runs': runs, 'evidence': evidence,
            'independent_test_used_for_selection': False,
        }
    if not output:
        raise ValueError('No clean monitoring trajectories found.')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', help='Calculate from clean runs generated in this project.')
    parser.add_argument('--datasets', nargs='+', choices=tuple(LABELS), default=list(LABELS))
    parser.add_argument('--clients', nargs='+', type=int, choices=(10, 20), default=[10, 20])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 47)))
    parser.add_argument('--data-root', default='data/processed')
    parser.add_argument('--output-root', default='results')
    parser.add_argument('--margin', type=float, default=0.000001)
    parser.add_argument('--output', default='results/mpdd_thresholds.json')
    parser.add_argument('--execute', action='store_true', help='Train clean models before calculating thresholds.')
    parser.add_argument('--resume', action='store_true')
    for name, default, datatype in [('rounds', 12, int), ('local-epochs', 40, int),
                                     ('batch-size', 500, int), ('learning-rate', .01, float),
                                     ('fit-verbose', 0, int)]:
        parser.add_argument('--' + name, type=datatype, default=default)
    parser.add_argument('--trim-batches', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.margin) or args.margin <= 0:
        parser.error('--margin must be positive and finite.')
    if args.rounds < 4 and args.results is None:
        parser.error('At least four rounds are needed for two warmup and two confirmation rounds.')
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite calibration output: {target}')
    roots = args.results
    if roots is None:
        overrides = {key: getattr(args, key) for key in
                     ('rounds', 'local_epochs', 'batch_size', 'learning_rate', 'fit_verbose', 'trim_batches')}
        overrides['save_model'] = False
        override_path = Path(args.output_root) / 'calibration_overrides.json'
        if override_path.exists() and read_json(override_path) != overrides:
            raise FileExistsError('Calibration settings changed; choose a new --output-root.')
        write_json(override_path, overrides)
        settings = ['1-out-of-10' if count == 10 else '4-out-of-20' for count in args.clients]
        command = ['--preset', 'calibration', '--datasets', *args.datasets, '--settings', *settings,
                   '--seeds', *[str(seed) for seed in args.seeds], '--data-root', args.data_root,
                   '--output-root', args.output_root, '--overrides', str(override_path)]
        if args.execute:
            command.append('--execute')
        if args.resume:
            command.append('--resume')
        code = run_batch(command)
        if code:
            return code
        if not args.execute:
            print('No training or threshold calculation performed; add --execute to run the plan.')
            return 0
        roots = [Path(args.output_root) / 'calibration']
    output = calibrate(roots, args.margin)
    if args.results is None:
        for dataset in args.datasets:
            if (dataset not in output or output[dataset]['client_counts'] != sorted(set(args.clients))
                    or output[dataset]['seeds'] != sorted(set(args.seeds))):
                raise ValueError('Completed calibration coverage differs from the requested plan.')
    write_json(target, {'schema': 'padae-mpdd-calibration-v1', 'datasets': output})
    for dataset, record in output.items():
        print(f"{dataset}: tau_R={record['tau_r']:.17g}; clean trajectories={record['trajectories']}")
    print(f'Calibration saved to {target}; load it with --calibration-file {target}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
