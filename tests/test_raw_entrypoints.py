from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from configuration import LABELS, RunConfig
from data_process import load_data
from preprocess.prepare import CIC_FILES
from preprocess.schema import CIC_COLUMNS, UNSW_COLUMNS
from utils import read_json


@pytest.mark.parametrize('dataset', ['UNSW-NB15', 'CIC-IDS2017'])
def test_raw_csv_entrypoint_builds_fresh_partitions(tmp_path, dataset):
    raw = tmp_path / 'raw' / dataset
    raw.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    if dataset == 'UNSW-NB15':
        labels = np.repeat(LABELS[dataset], 400)
        frame = pd.DataFrame({name: rng.normal(size=len(labels)) for name in UNSW_COLUMNS})
        frame['srcip'] = '192.0.2.1'
        frame['dstip'] = '198.51.100.2'
        frame['proto'] = np.resize(['tcp', 'udp', 'icmp'], len(labels))
        frame['state'] = np.resize(['fin', 'con'], len(labels))
        frame['service'] = np.resize(['http', '-', 'dns'], len(labels))
        frame['attack_cat'] = np.where(labels == 'normal', '', labels)
        frame['label'] = (labels != 'normal').astype(int)
        for index, positions in enumerate(np.array_split(np.arange(len(frame)), 4), 1):
            frame.iloc[positions].to_csv(raw / f'UNSW-NB15_{index}.csv', index=False, header=False)
        script = 'preprocess_unsw_nb15.py'
    else:
        groups = {'ddos': ['DDoS'], 'portscan': ['BENIGN', 'PortScan'], 'dos_hulk': ['DoS Hulk']}
        for key, classes in groups.items():
            labels = np.repeat(classes, 400)
            frame = pd.DataFrame({name: rng.normal(size=len(labels)) for name in CIC_COLUMNS})
            frame['Label'] = labels
            frame.to_csv(raw / CIC_FILES[key], index=False)
        script = 'Preprocess_cicids2017.py'
    processed = tmp_path / 'processed'
    command = [sys.executable, '-B', str(Path(__file__).parents[1] / 'preprocess' / script),
               '--raw-root', str(tmp_path / 'raw'), '--output-root', str(processed)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    for count, setting in [(10, '1-out-of-10'), (20, '4-out-of-20')]:
        directory = processed / dataset / f'clients_{count}'
        metadata = read_json(directory / 'preprocessing_manifest.json')
        assert metadata['held_out_used_for_fitting'] is False
        assert metadata['K'] == count
        assert len(metadata['files']) == 2 * count + 2
        cfg = RunConfig(dataset=dataset, client_setting=setting, data_root=str(processed),
                        batch_size=5).resolved()
        data = load_data(cfg)
        assert data.test.raw_rows == data.test.used_rows
        assert len(data.train) == count and len(data.local_test) == count
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert 'Refusing to overwrite' in rerun.stderr


@pytest.mark.parametrize('entry', [
    'main.py', 'run_experiments.py', 'summarize_results.py', 'calibrate_mpdd.py',
    'benchmark.py', 'preprocess/prepare.py',
    'preprocess/preprocess_unsw_nb15.py', 'preprocess/Preprocess_cicids2017.py',
])
def test_cli_help_entrypoint(entry):
    root = Path(__file__).parents[1]
    run = subprocess.run([sys.executable, '-B', str(root / entry), '--help'],
                         cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
