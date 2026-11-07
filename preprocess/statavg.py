"""Sample-count-weighted global moments from local training data."""
from __future__ import annotations
import numpy as np


def local_statistics(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Local training matrix must be finite and nonempty.')
    return {'count': len(a), 'mean': a.mean(axis=0), 'variance': a.var(axis=0, ddof=0)}


def aggregate_statistics(records, epsilon=1e-12):
    if not records:
        raise ValueError('No local training statistics.')
    counts = np.array([r['count'] for r in records], dtype=np.float64)
    means = np.stack([r['mean'] for r in records]); variances = np.stack([r['variance'] for r in records])
    if (np.any(counts <= 0) or not np.isfinite(counts).all()
            or not np.isfinite(means).all() or not np.isfinite(variances).all()
            or np.any(variances < -epsilon)):
        raise ValueError('Invalid local statistics.')
    weights = counts / counts.sum()
    mean = np.sum(weights[:, None] * means, axis=0)
    variance = np.sum(weights[:, None] * (np.maximum(variances, 0) + (means-mean)**2), axis=0)
    std = np.sqrt(np.maximum(variance, 0)); zero = std <= epsilon
    scale = std.copy(); scale[zero] = 1.0
    return {'mean': mean, 'variance': variance, 'std': std, 'scale': scale,
            'zero_variance': zero, 'count': int(counts.sum()), 'epsilon': epsilon}


def normalize(values, statistics):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != len(statistics['mean']) or not np.isfinite(a).all():
        raise ValueError('Invalid matrix for normalization.')
    result = (a-statistics['mean']) / statistics['scale']
    result[:, statistics['zero_variance']] = 0.0
    if not np.isfinite(result).all():
        raise ValueError('Normalization produced nonfinite values.')
    return result.astype(np.float32)
