"""Load fixed, preprocessed partitions without resplitting by training seed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from configuration import LABELS
from utils import digest, read_json, sha256_file


@dataclass
class PartitionData:
    x: np.ndarray
    y: np.ndarray
    features: tuple
    raw_rows: int
    used_rows: int


@dataclass
class DataBundle:
    train: list
    local_test: list
    validation: PartitionData
    test: Optional[PartitionData]
    manifest: dict


def read_partition(path, labels, features=None, batch_size=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required fixed partition not found: {path}')
    with path.open('r', encoding='utf-8-sig') as handle:
        import csv
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise ValueError(f'Duplicate columns in {path.name}.')
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty or any(label not in frame for label in labels):
        raise ValueError(f'Empty data or missing label columns: {path.name}')
    columns = tuple(c for c in frame if c not in labels)
    if any(c.lower() in ('attack_cat', 'label', 'srcip', 'dstip') for c in columns):
        raise ValueError(f'Raw identifier or target columns remain in {path.name}.')
    if features is not None and columns != tuple(features):
        raise ValueError(f'Feature order mismatch: {path.name}')
    if not columns:
        raise ValueError('The input feature set is empty.')
    x = frame.loc[:, columns].to_numpy(dtype=np.float32)
    y = frame.loc[:, list(labels)].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'Nonfinite preprocessed data in {path.name}.')
    if not np.all((y == 0) | (y == 1)) or not np.all(y.sum(axis=1) == 1):
        raise ValueError(f'Invalid one-hot targets in {path.name}.')
    used = len(frame)
    if batch_size is not None:
        used = used // batch_size * batch_size
        if used == 0:
            raise ValueError(f'{path.name} has fewer rows than batch_size={batch_size}.')
    return PartitionData(x[:used], y[:used], columns, len(frame), used)


def load_data(config):
    directory = config.client_dir; labels = LABELS[config.dataset]
    if not directory.is_dir():
        raise FileNotFoundError(f'Data directory not found: {directory}')
    batch = config.batch_size if config.trim_batches else None
    train, local_test, rows = [], [], []
    feature_names = None
    def load(name, trimming):
        nonlocal feature_names
        path = directory / name
        data = read_partition(path, labels, feature_names, trimming)
        if feature_names is None:
            feature_names = data.features
        rows.append({'name': name, 'sha256': sha256_file(path), 'bytes': path.stat().st_size,
                     'raw_rows': data.raw_rows, 'used_rows': data.used_rows})
        return data
    for client_id in range(config.K):
        train.append(load(f'client_{client_id+1:02d}_train.csv', batch))
        if not config.monitor_only:
            local_test.append(load(f'client_{client_id+1:02d}_test.csv', batch))
    validation = load('Val.csv', batch)
    test = None if config.monitor_only else load('Test.csv', None)
    manifest = {'dataset': config.dataset, 'K': config.K, 'features': list(feature_names),
                'class_order': list(labels), 'files': rows,
                'preprocessing_evidence_present': (directory / 'statavg_global_statistics.csv').exists()}
    manifest['fingerprint'] = digest({k: v for k, v in manifest.items()
                                      if k != 'preprocessing_evidence_present'})
    if config.expected_data_manifest:
        expected = read_json(config.expected_data_manifest)
        expected = expected.get(f'{config.dataset}|K{config.K}', expected)
        expected_files = expected.get('files', expected)
        if isinstance(expected_files, list):
            expected_files = {r['name']: r['sha256'] for r in expected_files}
        actual = {r['name']: r['sha256'] for r in rows}
        required = set(actual)
        if not required.issubset(expected_files):
            raise ValueError('Expected data manifest does not cover every used partition.')
        mismatch = [name for name in actual if actual[name] != expected_files[name]]
        if mismatch:
            raise ValueError(f'Data do not match reference hashes: {mismatch}')
    if config.calibration_file:
        from calibration_io import validate_calibration_data
        validate_calibration_data(config.calibration_file, config.dataset, config.K, rows)
    return DataBundle(train, local_test, validation, test, manifest)
