"""Load calibrated MPDD thresholds and verify their data coverage."""
from __future__ import annotations

import math
from pathlib import Path
from utils import read_json, sha256_file


def load_threshold(path, dataset, client_count=None):
    path = Path(path)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError('The calibration file must contain a JSON object.')
    records = payload.get('datasets', payload)
    if dataset not in records or not isinstance(records[dataset], dict):
        raise ValueError(f'No calibrated threshold for {dataset}.')
    record = records[dataset]
    value = record.get('tau_r')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('The calibrated tau_r must be numeric.')
    if not math.isfinite(value) or value <= 1.0:
        raise ValueError('The calibrated tau_r must be finite and greater than 1.')
    if client_count is not None and 'client_counts' in record:
        if client_count not in record['client_counts']:
            raise ValueError(f'Calibration does not cover {dataset} with K={client_count}.')
    return record, sha256_file(path)


def validate_calibration_data(path, dataset, client_count, data_rows):
    record, _ = load_threshold(path, dataset, client_count)
    coverage = record.get('training_data')
    if coverage is None:
        return
    expected = coverage.get(str(client_count))
    if not expected:
        raise ValueError('Calibration contains no training-data coverage for this client count.')
    actual = {row['name']: row['sha256'] for row in data_rows}
    mismatch = [name for name, checksum in expected.items() if actual.get(name) != checksum]
    if mismatch:
        raise ValueError(f'Calibration data do not match the current partitions: {mismatch}')
