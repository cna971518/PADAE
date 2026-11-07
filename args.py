"""Command-line configuration without importing TensorFlow."""
from __future__ import annotations

import argparse
from dataclasses import fields
from configuration import LABELS, METHODS, SCENARIOS, SETTINGS, RunConfig
from utils import read_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Run one fixed-partition PADAE experiment.')
    p.add_argument('--config', help='JSON object with RunConfig fields.')
    for name, choices in [('dataset', LABELS), ('client-setting', SETTINGS),
                          ('scenario', SCENARIOS), ('method', METHODS)]:
        p.add_argument('--'+name, choices=tuple(choices), default=None)
    for name in ('seed', 'rounds', 'local-epochs', 'batch-size', 'warmup-rounds',
                 'confirmation-rounds', 'mqv-confirmation-rounds', 'fit-verbose'):
        p.add_argument('--'+name, type=int, default=None)
    for name in ('learning-rate', 'tau-mqv', 'tau-r', 'beta', 'lam', 'trim-ratio'):
        p.add_argument('--'+name, type=float, default=None)
    for name in ('data-root', 'output', 'expected-data-manifest', 'calibration-file', 'tag'):
        p.add_argument('--'+name, default=None)
    p.add_argument('--cma-rounding', choices=('ceil',), default=None)
    for name in ('minimum-survivors', 'trim-batches', 'save-model', 'record-pairs', 'monitor-only'):
        p.add_argument('--'+name, action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--check-only', action='store_true', help='Check configuration and data; do not train.')
    p.add_argument('--check-report', help='Optional JSON path for the data check.')
    args = p.parse_args(argv)
    valid = {f.name for f in fields(RunConfig)}
    settings = read_json(args.config) if args.config else {}
    if not isinstance(settings, dict) or set(settings)-valid:
        p.error(f'Unknown RunConfig fields: {sorted(set(settings)-valid)}')
    settings.update({k:v for k,v in vars(args).items() if k in valid and v is not None})
    if args.calibration_file is not None:
        settings['calibration_sha256'] = None
        if args.tau_r is None:
            settings['tau_r'] = None
    try:
        config = RunConfig(**settings).resolved()
    except (TypeError, ValueError) as error:
        p.error(str(error))
    return config, args.check_only, args.check_report
