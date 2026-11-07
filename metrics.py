"""Model-quality and independent-test metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def validate_probabilities(y, predictions):
    y = np.asarray(y); predictions = np.asarray(predictions)
    if y.ndim != 2 or y.shape != predictions.shape or not len(y) or y.shape[1] < 2:
        raise ValueError('Predictions must match nonempty multiclass one-hot labels.')
    if not np.all((y == 0) | (y == 1)) or not np.all(y.sum(axis=1) == 1):
        raise ValueError('Expected one-hot labels.')
    if not np.isfinite(predictions).all() or np.any(predictions < 0) or np.any(predictions > 1):
        raise ValueError('Predictions must be finite probabilities.')
    if not np.allclose(predictions.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Softmax probabilities must sum to one.')
    return y, predictions


def mqv_score(y, predictions):
    y, predictions = validate_probabilities(y, predictions)
    fpr, tpr, _ = roc_curve(y.ravel(), predictions.ravel())
    return float(np.max(np.abs(tpr - fpr)))


def classify(y, predictions, class_names):
    y, predictions = validate_probabilities(y, predictions)
    if len(class_names) != y.shape[1] or len(set(class_names)) != len(class_names):
        raise ValueError('class_names must match the unique output-class order.')
    cm = confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1),
                          labels=np.arange(y.shape[1]))
    rows = []
    for index, name in enumerate(class_names):
        tp = int(cm[index, index]); fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp); tn = int(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append({'class': name, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                     'support': tp + fn, 'precision': 100 * precision,
                     'recall': 100 * recall, 'f1': 100 * f1})
    result = {'accuracy': 100 * float(np.trace(cm) / cm.sum()),
              'macro_f1': float(np.mean([r['f1'] for r in rows])),
              'samples': int(cm.sum())}
    return result, rows, cm
