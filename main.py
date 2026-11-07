"""Run one experiment or verify its fixed data partitions."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import traceback


def main(argv=None):
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_CUDNN_DETERMINISM', '1')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from args import parse_args
    from data_process import load_data
    from utils import write_json
    config, check_only, check_report = parse_args(argv)
    if check_only:
        data = load_data(config)
        report = {'status': 'data_check_passed', 'config': config.as_dict(), 'data': data.manifest,
                  'training_executed': False}
        if check_report:
            write_json(check_report, report)
        print(f'Data check passed: {config.dataset}, K={config.K}, '
              f'features={len(data.validation.features)}, validation={data.validation.used_rows}, '
              f'independent_test={data.test.used_rows if data.test is not None else 'not_loaded'}')
        return 0
    if os.environ.get('PYTHONHASHSEED') != str(config.seed):
        if argv is not None:
            raise RuntimeError('Set PYTHONHASHSEED before starting a programmatic training process.')
        env = dict(os.environ, PYTHONHASHSEED=str(config.seed))
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    import importlib.util
    if importlib.util.find_spec('tensorflow') is None:
        raise RuntimeError('TensorFlow is not installed. Use the documented Python 3.9 / TensorFlow 2.10 environment.')
    from server import FederatedRunner
    runner = FederatedRunner(config)
    try:
        result = runner.run()
    except Exception as error:
        out = Path(config.output)
        if runner.output_owned:
            write_json(out / 'FAILED.json', {'error_type': type(error).__name__, 'message': str(error),
                                            'completed': False})
            (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    if config.monitor_only:
        print(f"status={result['status']} clean monitoring rounds={result['completed_rounds']}")
    else:
        print(f"status={result['status']} ACC={result['accuracy']:.6f} Macro-F1={result['macro_f1']:.6f}")
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
