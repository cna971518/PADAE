"""Fixed, local poisoning simulations for the manuscript experiments."""
from __future__ import annotations

import numpy as np
from configuration import LABELS

PDT_PAIRS = {
    'UNSW-NB15': [('dos', 'exploits', list(range(0, 8))),
                 ('fuzzers', 'generic', list(range(8, 16))),
                 ('normal', 'reconnaissance', list(range(16, 25)))],
    'CIC-IDS2017': [('BENIGN', 'DoS_Hulk', list(range(0, 10)) + list(range(35, 45))),
                   ('DDoS', 'PortScan', list(range(20, 30)) + list(range(51, 61)))],
}


def validate_xy(x, y, dataset):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.shape != (len(x), len(LABELS[dataset])) or not len(x):
        raise ValueError('Invalid feature or label shape.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite features or labels.')
    if not np.all((y == 0) | (y == 1)) or not np.all(y.sum(axis=1) == 1):
        raise ValueError('One-hot labels are required.')
    return x, y


def sample_count(size, ratio):
    if not 0 <= ratio <= 1:
        raise ValueError('Poisoning ratio must be in [0,1].')
    return 0 if ratio == 0 or size == 0 else min(size, max(1, int(size * ratio)))


def partial_data_tampering(x, y, dataset, ratio, seed):
    x, y = validate_xy(x, y, dataset)
    output = x.copy(); original = x.copy(); classes = y.argmax(axis=1)
    records = []
    for pair_id, (a, b, features) in enumerate(PDT_PAIRS[dataset]):
        if max(features) >= x.shape[1]:
            raise ValueError(f'PDT requires feature index {max(features)}; got {x.shape[1]} features.')
        ia = np.flatnonzero(classes == LABELS[dataset].index(a))
        ib = np.flatnonzero(classes == LABELS[dataset].index(b))
        na, nb = sample_count(len(ia), ratio), sample_count(len(ib), ratio)
        if not na or not nb:
            records.append({'pair': [a, b], 'selected_a': 0, 'selected_b': 0,
                            'reason': 'zero_budget_or_missing_class'})
            continue
        rng = np.random.default_rng(seed + pair_id)
        sa = rng.choice(ia, na, replace=False); sb = rng.choice(ib, nb, replace=False)
        da = rng.choice(ib, na, replace=True); db = rng.choice(ia, nb, replace=True)
        donor_a = original[np.ix_(da, features)].copy()
        donor_b = original[np.ix_(db, features)].copy()
        output[np.ix_(sa, features)] = donor_a
        output[np.ix_(sb, features)] = donor_b
        records.append({'pair': [a, b], 'selected_a': na, 'selected_b': nb,
                        'features': features, 'changed_values': int(
                            np.count_nonzero(output[np.ix_(sa, features)] != original[np.ix_(sa, features)])
                            + np.count_nonzero(output[np.ix_(sb, features)] != original[np.ix_(sb, features)]))})
    return output, y.copy(), records


def label_flipping(x, y, dataset, ratio, seed):
    x, y = validate_xy(x, y, dataset)
    output = y.copy(); classes = y.argmax(axis=1); rng = np.random.default_rng(seed)
    records = []
    for a, b, _ in PDT_PAIRS[dataset]:
        ca, cb = LABELS[dataset].index(a), LABELS[dataset].index(b)
        ia, ib = np.flatnonzero(classes == ca), np.flatnonzero(classes == cb)
        na, nb = sample_count(len(ia), ratio), sample_count(len(ib), ratio)
        if not na or not nb:
            records.append({'pair': [a, b], 'flipped': 0}); continue
        sa, sb = rng.choice(ia, na, replace=False), rng.choice(ib, nb, replace=False)
        output[sa] = 0; output[sa, cb] = 1
        output[sb] = 0; output[sb, ca] = 1
        records.append({'pair': [a, b], 'flipped': na + nb})
    return x.copy(), output, records


def random_weights(weights, seed, scale=1.0):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Random-weight scale must be positive.')
    rng = np.random.default_rng(seed)
    return [rng.normal(0, scale, size=np.asarray(w).shape).astype(np.asarray(w).dtype)
            for w in weights]


def training_data(x, y, dataset, scenario, seed):
    if scenario.startswith('pdt_'):
        return partial_data_tampering(x, y, dataset, 1.0 if scenario == 'pdt_1p0' else .5, seed)
    if scenario.startswith('lf_'):
        return label_flipping(x, y, dataset, 1.0 if scenario == 'lf_1p0' else .5, seed)
    if scenario in ('clean', 'rw_1p0'):
        return x, y, []
    raise ValueError(f'Unsupported scenario: {scenario}')
