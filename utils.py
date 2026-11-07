"""Output, hashing, and runtime metadata."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def weights_hash(weights):
    h = hashlib.sha256()
    for tensor in weights:
        a = np.ascontiguousarray(tensor)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(json_ready(v)) if isinstance(v, (list, tuple, dict, np.ndarray))
                             else json_ready(v) for k, v in row.items()})


def runtime_metadata():
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'imbalanced-learn',
                 'tensorflow', 'tensorflow-gpu', 'keras', 'joblib', 'psutil'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {'python': platform.python_version(), 'system': platform.system(),
            'architecture': platform.machine(), 'processor': platform.processor(),
            'host_fingerprint': hashlib.sha256(platform.node().encode()).hexdigest()[:16],
            'packages': packages}
    tf = sys.modules.get('tensorflow')
    if tf is not None:
        info['tensorflow_actual'] = tf.__version__
        info['devices'] = [{'name': d.name, 'type': d.device_type,
                            'details': tf.config.experimental.get_device_details(d)}
                           for d in tf.config.list_physical_devices()]
    info['environment'] = {k: os.environ.get(k) for k in
                           ('TF_DETERMINISTIC_OPS', 'TF_CUDNN_DETERMINISM', 'TF_ENABLE_ONEDNN_OPTS',
                            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
                            'TF_NUM_INTEROP_THREADS')}
    return info


def source_hashes(root):
    root = Path(root)
    ignored = {'venv', 'env', '__pycache__', 'data', 'results'}
    files = []
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in ignored and not name.startswith('.'))
        files.extend(Path(directory) / name for name in names if name.endswith('.py'))
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files)}
