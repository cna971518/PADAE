from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import calibrate_mpdd
from args import parse_args
from calibration_io import load_threshold
from calibrate_mpdd import calibrate, calibration_candidates
from configuration import RunConfig
from conftest import FakeModel
from data_process import load_data
from run_experiments import make_plan
from server import FederatedRunner
from utils import read_json, sha256_file, source_hashes, write_json


class CleanMonitoringModel(FakeModel):
    def fit(self, x, y, **kwargs):
        self.fits += 1
        client = int(self.name.split('_')[-1]) - 1
        direction = [1., .03 * client] if client < 7 else [.02 * client, 1.]
        self.weights[0] += np.broadcast_to(np.asarray(direction, dtype=np.float32), self.weights[0].shape)

    def predict(self, x, **kwargs):
        raise AssertionError('Clean MPDD monitoring must not evaluate held-out data.')


def write_monitor_run(root, data_root, seed=42, count=10):
    setting = '1-out-of-10' if count == 10 else '4-out-of-20'
    cfg = RunConfig(dataset='UNSW-NB15', client_setting=setting, scenario='clean',
                    method='mpdd_only', monitor_only=True, seed=seed, rounds=4,
                    local_epochs=1, batch_size=10, data_root=str(data_root),
                    output=str(root / f'clean_K{count}_s{seed}'), save_model=False).resolved()
    result = FederatedRunner(cfg, model_factory=CleanMonitoringModel,
                             seed_function=lambda seed: None).run()
    return cfg, result


def save_thresholds(path, records):
    write_json(path, {'schema': 'padae-mpdd-calibration-v1', 'datasets': records})


def refresh_done(directory):
    write_json(directory / 'DONE.json', {'output_hashes': {
        p.name: sha256_file(p) for p in directory.iterdir()
        if p.is_file() and p.name != 'DONE.json'}})


def test_clean_monitoring_does_not_load_test_partitions(fixed_data, tmp_path):
    data_root = fixed_data()
    folder = data_root / 'UNSW-NB15' / 'clients_10'
    for path in list(folder.glob('*_test.csv')) + [folder / 'Test.csv']:
        path.unlink()
    cfg, result = write_monitor_run(tmp_path / 'calibration', data_root)
    manifest = read_json(Path(cfg.output) / 'data_manifest.json')
    assert len(manifest['files']) == 11
    assert not any('test' in r['name'].lower() for r in manifest['files'])
    assert result['accuracy'] is None and result['macro_f1'] is None
    assert result['independent_test_evaluated'] is False
    assert result['removed_benign'] == 0 and result['completed_rounds'] == 4
    assert not (Path(cfg.output) / 'confusion_matrix.csv').exists()


def test_calibration_and_loading_from_new_runs(fixed_data, tmp_path):
    data_root = fixed_data()
    root = tmp_path / 'clean'
    for seed in (42, 43):
        cfg, result = write_monitor_run(root, data_root, seed)
    report = calibrate([root], 1e-6)
    entry = report['UNSW-NB15']
    assert entry['trajectories'] == 2 and entry['seeds'] == [42, 43]
    assert entry['tau_r'] == pytest.approx(entry['max_two_round_min_R'] + 1e-6)
    assert entry['tau_r'] > entry['max_two_round_min_R']
    assert all(row['rounds'] == [3, 4] for row in entry['evidence'])
    artifact = tmp_path / 'thresholds.json'
    save_thresholds(artifact, report)
    normal = replace(cfg, output=str(tmp_path / 'normal'), monitor_only=False, method='padae',
                     seed=100, tau_r=None, calibration_file=str(artifact)).resolved()
    data = load_data(normal)
    assert normal.tau_r == entry['tau_r'] and data.test is not None
    assert normal.calibration_sha256 == sha256_file(artifact)


def test_calibration_rejects_changed_training_data(fixed_data, tmp_path):
    data_root = fixed_data()
    cfg, _ = write_monitor_run(tmp_path / 'clean', data_root)
    artifact = tmp_path / 'thresholds.json'
    save_thresholds(artifact, calibrate([tmp_path / 'clean'], 1e-6))
    evaluation = replace(cfg, monitor_only=False, method='padae', tau_r=None,
                         calibration_file=str(artifact)).resolved()
    path = evaluation.client_dir / 'client_01_train.csv'
    frame = pd.read_csv(path)
    frame.iloc[0, 0] += .5
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Calibration data do not match'):
        load_data(evaluation)


def test_calibration_file_mutation_is_detected(fixed_data, tmp_path):
    data_root = fixed_data()
    cfg, _ = write_monitor_run(tmp_path / 'clean', data_root)
    artifact = tmp_path / 'thresholds.json'
    records = calibrate([tmp_path / 'clean'], 1e-6)
    save_thresholds(artifact, records)
    evaluation = replace(cfg, monitor_only=False, method='padae', tau_r=None,
                         calibration_file=str(artifact)).resolved()
    records['UNSW-NB15']['tau_r'] += .1
    save_thresholds(artifact, records)
    with pytest.raises(ValueError, match='calibration file changed'):
        evaluation.resolved()


def test_calibration_cli_overrides_config_threshold(tmp_path):
    artifact = tmp_path / 'thresholds.json'
    save_thresholds(artifact, {'UNSW-NB15': {'tau_r': 1.9, 'client_counts': [10]}})
    config_file = tmp_path / 'config.json'
    write_json(config_file, RunConfig().resolved().as_dict())
    cfg, _, _ = parse_args(['--config', str(config_file), '--calibration-file', str(artifact)])
    assert cfg.tau_r == 1.9
    with pytest.raises(SystemExit):
        parse_args(['--calibration-file', str(artifact), '--tau-r', '2.1'])


def test_calibration_rejects_uncovered_dataset_or_count(tmp_path):
    path = tmp_path / 'thresholds.json'
    save_thresholds(path, {'UNSW-NB15': {'tau_r': 1.9, 'client_counts': [10]}})
    with pytest.raises(ValueError, match='No calibrated threshold'):
        load_threshold(path, 'CIC-IDS2017', 10)
    with pytest.raises(ValueError, match='does not cover'):
        load_threshold(path, 'UNSW-NB15', 20)


def test_batch_uses_calibrated_threshold(tmp_path):
    artifact = tmp_path / 'thresholds.json'
    save_thresholds(artifact, {'UNSW-NB15': {'tau_r': 1.7, 'client_counts': [10]}})
    plan = make_plan('ablation', datasets=['UNSW-NB15'], seeds=[100],
                     calibration_file=str(artifact))
    assert len(plan) == 40
    assert all(row['config']['tau_r'] == 1.7 for row in plan)
    assert all(row['config']['calibration_sha256'] for row in plan)
    with pytest.raises(ValueError, match='does not load'):
        make_plan('calibration', calibration_file=str(artifact))


def test_calibration_rejects_partial_trajectory(fixed_data, tmp_path):
    data_root = fixed_data()
    cfg, _ = write_monitor_run(tmp_path / 'clean', data_root)
    directory = Path(cfg.output)
    rows = pd.read_csv(directory / 'cluster_round_summary.csv')
    rows.iloc[:-1].to_csv(directory / 'cluster_round_summary.csv', index=False)
    refresh_done(directory)
    with pytest.raises(ValueError, match='round records are incomplete'):
        calibrate([tmp_path / 'clean'], 1e-6)


def test_calibration_rejects_invalid_or_zero_margin(tmp_path):
    for margin in (0., -1., float('inf'), float('nan')):
        with pytest.raises(ValueError, match='margin'):
            calibrate([tmp_path], margin)


def test_calibration_rejects_duplicate_rounds():
    row = {'round': 3, 'episode': 1, 'warmup': False, 'structural_valid': True,
           'R': 1.2, 'suspicious_ids': [0]}
    with pytest.raises(ValueError, match='Duplicate round'):
        calibration_candidates([row, row])


def test_calibration_excludes_gaps_invalid_splits_and_identity_changes():
    def row(t, suspicious, **extra):
        return dict(round=t, episode=1, warmup=False, structural_valid=True,
                    R=1.5, suspicious_ids=suspicious, **extra)
    rows = [row(3, [0]), row(4, [1]), row(6, [1])]
    assert calibration_candidates(rows) == []
    rows = [row(3, [0]), row(4, [0]), row(5, [0])]
    rows[1]['structural_valid'] = False
    assert calibration_candidates(rows) == []


def test_calibration_dry_run_creates_plan_without_threshold(tmp_path):
    target = tmp_path / 'thresholds.json'
    code = calibrate_mpdd.main(['--datasets', 'UNSW-NB15', '--clients', '10',
                               '--seeds', '42', '--output-root', str(tmp_path),
                               '--output', str(target)])
    assert code == 0 and not target.exists()
    plan = read_json(tmp_path / 'plan_calibration.json')
    assert plan['planned_runs'] == 1
    assert plan['runs'][0]['config']['monitor_only'] is True
    assert plan['runs'][0]['config']['expected_data_manifest'] is None


def test_calibration_end_to_end_control_flow(fixed_data, tmp_path, monkeypatch):
    data_root = fixed_data()
    output = tmp_path / 'runs'
    def execute(argv):
        plan = make_plan('calibration', datasets=['UNSW-NB15'], settings=['1-out-of-10'],
                         seeds=[42, 43], data_root=str(data_root), output_root=str(output),
                         overrides={'rounds': 4, 'local_epochs': 1, 'batch_size': 10, 'save_model': False})
        for item in plan:
            FederatedRunner(RunConfig(**item['config']), model_factory=CleanMonitoringModel,
                            seed_function=lambda seed: None).run()
        return 0
    monkeypatch.setattr(calibrate_mpdd, 'run_batch', execute)
    target = tmp_path / 'thresholds.json'
    code = calibrate_mpdd.main(['--datasets', 'UNSW-NB15', '--clients', '10',
                               '--seeds', '42', '43', '--rounds', '4', '--local-epochs', '1',
                               '--batch-size', '10', '--data-root', str(data_root),
                               '--output-root', str(output), '--output', str(target), '--execute'])
    assert code == 0
    entry, _ = load_threshold(target, 'UNSW-NB15', 10)
    assert entry['trajectories'] == 2 and entry['tau_r'] > 1


def test_source_hashes_ignore_virtualenv_and_data(tmp_path):
    for name in ['main.py', 'preprocess/prepare.py', '.venv/lib/thirdparty.py', 'results/mock.py', 'data/x.py']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x = 1\n')
    assert set(source_hashes(tmp_path)) == {'main.py', 'preprocess/prepare.py'}


def test_cma_floor_cannot_silently_change_protocol():
    with pytest.raises(ValueError, match='aggregation'):
        RunConfig(cma_rounding='floor').resolved()
