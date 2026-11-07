"""Model aggregation and reference defense methods."""
from __future__ import annotations

import math
import numpy as np


def aggregate_models(models, weights):
    w = np.asarray(weights, dtype=np.float64)
    if not models or w.shape != (len(models),) or not np.isfinite(w).all():
        raise ValueError('Model/weight coverage mismatch.')
    if np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-12):
        raise ValueError('Aggregation weights must be nonnegative and sum to one.')
    if not models[0] or any(len(m) != len(models[0]) for m in models):
        raise ValueError('Inconsistent tensor counts.')
    result = []
    for j, reference in enumerate(models[0]):
        reference = np.asarray(reference)
        if reference.dtype.kind != 'f':
            raise TypeError('Model tensors must be floating point.')
        value = np.zeros(reference.shape, dtype=np.float64)
        for coefficient, model in zip(w, models):
            a = np.asarray(model[j], dtype=np.float64)
            if a.shape != reference.shape or not np.isfinite(a).all():
                raise ValueError('Incompatible or nonfinite model tensor.')
            value += coefficient * a
        result.append(value.astype(reference.dtype))
    return result


def cma_weights(ids, scores, beta, lam, rounding='ceil'):
    if not ids or not 0 < beta <= 1 or not 0 <= lam <= 1 or rounding != 'ceil':
        raise ValueError('Invalid CMA configuration.')
    q = np.asarray([scores[i] for i in ids], dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError('Invalid CMA scores.')
    order = np.argsort(-q, kind='stable')
    count = min(len(ids), max(1, int(math.ceil(beta * len(ids)))))
    weights = np.zeros(len(ids), dtype=np.float64)
    if count == len(ids):
        weights[:] = 1 / len(ids)
    else:
        weights[order[:count]] = lam / count
        weights[order[count:]] = (1 - lam) / (len(ids) - count)
    return weights


def fedavg_weights(sample_counts):
    counts = np.asarray(sample_counts, dtype=np.float64)
    if not len(counts) or np.any(counts <= 0) or not np.isfinite(counts).all():
        raise ValueError('Invalid effective training counts.')
    return counts / counts.sum()


def coordinate_aggregate(models, method, trim_ratio=.2):
    if not models or method not in ('median', 'trimmed_mean') or not 0 <= trim_ratio < .5:
        raise ValueError('Invalid coordinate aggregation configuration.')
    count = len(models); trim = math.floor(trim_ratio * count); output = []
    for j, reference in enumerate(models[0]):
        a = np.stack([np.asarray(m[j], dtype=np.float64) for m in models])
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite upload.')
        value = np.median(a, axis=0) if method == 'median' else np.sort(a, axis=0)[trim:count-trim].mean(axis=0)
        output.append(value.astype(np.asarray(reference).dtype))
    return output


def ddaba_weights(accuracies):
    """The recorded text_consistent_plus variant; see docs/IMPLEMENTATION.md."""
    acc = np.asarray(accuracies, dtype=np.float64)
    if acc.ndim != 1 or not acc.size or not np.isfinite(acc).all() or np.any((acc < 0) | (acc > 1)):
        raise ValueError('Validation accuracy must be a finite fraction in [0,1].')
    gaps = acc.max() - acc; mu = float(gaps.mean())
    if mu == 0:
        return np.full(len(acc), 1 / len(acc)), {'rule': 'text_consistent_plus', 'equal_accuracy': True}
    top = gaps <= np.log(10 / 9) * mu
    kept = gaps < (np.log(4) + 1.5 * np.log(3)) * mu
    nt, nr = int(top.sum()), int((kept & ~top).sum())
    weights = (2 * top.astype(float) + (kept & ~top).astype(float)) / (2 * nt + nr)
    return weights, {'rule': 'text_consistent_plus', 'top_count': nt, 'other_kept_count': nr,
                     'discarded_count': int((~kept).sum()), 'mean_accuracy_gap': mu}


def flatten(weights):
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in weights])


def positive_cosine(vectors, reference, eps=1e-12):
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(reference)
    return np.maximum(0, np.divide(vectors @ reference, denominator,
                                  out=np.zeros(len(vectors)), where=denominator > eps))


def middle_mean(scores):
    a = np.sort(np.asarray(scores, dtype=np.float64)); trim = len(a) // 4
    return float(a[trim:len(a)-trim].mean())


def trust_aggregate(models, global_weights, root_weights, method, seed, eps=1e-12):
    if method not in ('fltrust', 'flest') or not models:
        raise ValueError('Invalid trust aggregation request.')
    g = flatten(global_weights)
    updates = np.stack([flatten(m) - g for m in models]); root = flatten(root_weights) - g
    if not np.isfinite(updates).all() or not np.isfinite(root).all():
        raise ValueError('Nonfinite trust-aggregation update.')
    root_norm = float(np.linalg.norm(root)); norms = np.linalg.norm(updates, axis=1)
    trust_scores = positive_cosine(updates, root, eps)
    confidence = np.zeros(len(updates)); gamma = 1.0; fallback = None
    if method == 'flest':
        from sklearn.cluster import KMeans
        if len(updates) < 2 or np.unique(updates, axis=0).shape[0] < 2:
            confidence = positive_cosine(updates, updates.mean(axis=0), eps)
            fallback = 'identical_updates_single_reference'
        else:
            labels = KMeans(n_clusters=2, random_state=int(seed), n_init=10).fit_predict(updates)
            counts = np.bincount(labels, minlength=2)
            if counts[0] == counts[1]:
                confidence = trust_scores.copy(); fallback = 'equal_cluster_sizes_root_trust'
            else:
                reference = updates[labels == int(np.argmax(counts))].mean(axis=0)
                confidence = positive_cosine(updates, reference, eps)
        a, b = middle_mean(trust_scores), middle_mean(confidence)
        gamma = a / (a + b) if a + b > eps else .5
    trust = trust_scores.copy() if method == 'fltrust' else gamma * trust_scores + (1-gamma) * confidence
    trust[norms <= eps] = 0
    info = {'method': method, 'root_update_norm': root_norm, 'gamma': gamma, 'fallback': fallback,
            'trust_scores': trust_scores.tolist(), 'confidence_scores': confidence.tolist()}
    if root_norm <= eps or trust.sum() <= eps:
        info['fallback'] = 'no_global_update_zero_root_or_trust'
        return [np.array(a, copy=True) for a in global_weights], np.zeros(len(models)), info
    weights = trust / trust.sum()
    scales = np.divide(root_norm, norms, out=np.zeros(len(norms)), where=norms > eps)
    combined = g + np.sum(weights[:, None] * scales[:, None] * updates, axis=0)
    output = []; offset = 0
    for reference in global_weights:
        a = np.asarray(reference); size = a.size
        output.append(combined[offset:offset+size].reshape(a.shape).astype(a.dtype)); offset += size
    info['norm_scales'] = scales.tolist()
    return output, weights, info
