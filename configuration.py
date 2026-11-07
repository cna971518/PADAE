"""Fixed manuscript settings and explicit experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import math

LABELS = {
    'UNSW-NB15': ('dos', 'exploits', 'fuzzers', 'generic', 'normal', 'reconnaissance'),
    'CIC-IDS2017': ('BENIGN', 'DDoS', 'PortScan', 'DoS_Hulk'),
}
SETTINGS = {'1-out-of-10': (10, 1), '4-out-of-10': (10, 4), '4-out-of-20': (20, 4)}
THRESHOLDS = {'UNSW-NB15': (0.40, 1.0387229896327386),
              'CIC-IDS2017': (0.70, 1.1887096500465608)}
CMA_PARAMETERS = {'UNSW-NB15': (0.2, 0.5), 'CIC-IDS2017': (0.2, 0.9)}
SCENARIOS = ('clean', 'pdt_1p0', 'pdt_0p5', 'lf_1p0', 'lf_0p5', 'rw_1p0')
MODULES = {
    'uniform': (False, False, False), 'mqv_only': (True, False, False),
    'mpdd_only': (False, True, False), 'cma_only': (False, False, True),
    'mqv_mpdd': (True, True, False), 'mqv_cma': (True, False, True),
    'mpdd_cma': (False, True, True), 'padae': (True, True, True),
}
BASELINES = ('fedavg', 'median', 'trimmed_mean', 'ddaba', 'fltrust', 'flest')
METHODS = tuple(MODULES) + BASELINES


@dataclass(frozen=True)
class RunConfig:
    dataset: str = 'UNSW-NB15'
    client_setting: str = '4-out-of-10'
    scenario: str = 'clean'
    method: str = 'padae'
    seed: int = 100
    data_root: str = 'data/processed'
    output: str = 'results/run'
    rounds: int = 12
    local_epochs: int = 40
    batch_size: int = 500
    learning_rate: float = 0.01
    tau_mqv: Optional[float] = None
    tau_r: Optional[float] = None
    beta: Optional[float] = None
    lam: Optional[float] = None
    cma_rounding: str = 'ceil'
    trim_ratio: float = 0.2
    warmup_rounds: int = 2
    confirmation_rounds: int = 2
    mqv_confirmation_rounds: int = 2
    minimum_survivors: bool = True
    monitor_only: bool = False
    fit_verbose: int = 0
    trim_batches: bool = True
    save_model: bool = True
    record_pairs: bool = False
    expected_data_manifest: Optional[str] = None
    calibration_file: Optional[str] = None
    calibration_sha256: Optional[str] = None
    tag: str = 'experiment'

    def resolved(self):
        if self.dataset not in LABELS or self.client_setting not in SETTINGS:
            raise ValueError('Unsupported dataset or client setting.')
        if self.method not in METHODS or self.scenario not in SCENARIOS:
            raise ValueError('Unsupported method or scenario.')
        values = asdict(self)
        tm, tr = THRESHOLDS[self.dataset]
        if self.monitor_only:
            tr = 1.000001
            if self.calibration_file:
                raise ValueError('Clean monitoring must not load an existing calibration.')
        if self.calibration_file:
            from calibration_io import load_threshold
            record, checksum = load_threshold(self.calibration_file, self.dataset, self.K)
            if self.calibration_sha256 not in (None, checksum):
                raise ValueError('The calibration file changed after this run was configured.')
            tr = float(record['tau_r'])
            if self.tau_r is not None and self.tau_r != tr:
                raise ValueError('tau_r conflicts with the supplied calibration file.')
            values['calibration_sha256'] = checksum
            for key, expected in record.get('training_protocol', {}).items():
                if key in values and values[key] != expected:
                    raise ValueError(f'Calibration training protocol differs for {key}.')
        b, l = CMA_PARAMETERS[self.dataset]
        for key, default in [('tau_mqv', tm), ('tau_r', tr), ('beta', b), ('lam', l)]:
            if values[key] is None:
                values[key] = default
        result = RunConfig(**values)
        result.validate()
        return result

    def validate(self):
        for field in ('seed', 'rounds', 'local_epochs', 'batch_size', 'warmup_rounds',
                      'confirmation_rounds', 'mqv_confirmation_rounds', 'fit_verbose'):
            if type(getattr(self, field)) is not int:
                raise ValueError(f'{field} must be an integer.')
        for field in ('minimum_survivors', 'monitor_only', 'trim_batches', 'save_model', 'record_pairs'):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f'{field} must be a JSON boolean.')
        for field in ('data_root', 'output', 'tag'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f'{field} must be a nonempty string.')
        for field in ('rounds', 'local_epochs', 'batch_size'):
            if getattr(self, field) < 1:
                raise ValueError(f'{field} must be positive.')
        if not 0 <= self.seed < 2**32 - self.K:
            raise ValueError('seed is outside the supported RNG range.')
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError('learning_rate must be positive and finite.')
        if not 0 <= self.tau_mqv <= 1 or not math.isfinite(self.tau_r) or self.tau_r <= 1:
            raise ValueError('Invalid MQV or MPDD threshold.')
        if not 0 < self.beta <= 1 or not 0 <= self.lam <= 1:
            raise ValueError('Invalid CMA parameters.')
        if self.cma_rounding != 'ceil' or not 0 <= self.trim_ratio < .5:
            raise ValueError('Invalid aggregation configuration.')
        if self.warmup_rounds < 2 or min(self.confirmation_rounds, self.mqv_confirmation_rounds) < 2:
            raise ValueError('At least two warmup/confirmation rounds are required.')
        if self.fit_verbose not in (0, 1, 2):
            raise ValueError('fit_verbose must be 0, 1 or 2.')
        if self.calibration_file is not None and (not isinstance(self.calibration_file, str) or not self.calibration_file.strip()):
            raise ValueError('calibration_file must be a nonempty path.')
        if self.calibration_sha256 is not None and not self.calibration_file:
            raise ValueError('calibration_sha256 requires calibration_file.')
        if self.monitor_only and (self.scenario != 'clean' or self.method != 'mpdd_only'):
            raise ValueError('Calibration monitoring requires clean / mpdd_only.')

    @property
    def K(self):
        return SETTINGS[self.client_setting][0]

    @property
    def malicious_ids(self):
        return () if self.scenario == 'clean' else tuple(range(SETTINGS[self.client_setting][1]))

    @property
    def module_flags(self):
        return MODULES.get(self.method, (False, False, False))

    @property
    def client_dir(self):
        return Path(self.data_root) / self.dataset / f'clients_{self.K}'

    def as_dict(self):
        return asdict(self)
