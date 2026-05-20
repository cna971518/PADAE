# -*- coding: utf-8 -*-
"""
Poisoning attack utilities for PADAE federated learning experiments.

This file implements three poisoning attack scenarios:
1. Partial Data Tampering (PDT)
2. Label-flipping attack
3. Random weight attack

PDT setting:

UNSW-NB15:
- DoS & Exploits: features 0 ~ 7
- Fuzzers & Generic: features 8 ~ 15
- Normal & Reconnaissance: features 16 ~ 24

CIC-IDS2017:
- BENIGN & DoS_Hulk: features 0 ~ 9 and 35 ~ 44
- DDoS & PortScan: features 20 ~ 29 and 51 ~ 60

Label flipping setting:

UNSW-NB15:
- DoS <-> Exploits
- Fuzzers <-> Generic
- Normal <-> Reconnaissance

CIC-IDS2017:
- BENIGN <-> DoS_Hulk
- DDoS <-> PortScan

Note:
The labels are assumed to be One-Hot encoded after preprocessing.
PDT modifies only feature values and keeps labels unchanged.
Label flipping modifies only labels and keeps features unchanged.

Reproducibility:
- No server/FedAvg import is used here to avoid circular import.
- All random operations use numpy.random.default_rng(random_state).
- PDT pair-level sampling uses random_state + pair_id.
- Label flipping and random weight attack use the supplied random_state.

PDT formula:
    new_feature = (1 - alpha) * original_feature
                  + alpha * opposite_class_mean

where:
    tamper_ratio controls how many samples are tampered.
    alpha controls how strongly the selected samples are moved toward
    the opposite class distribution.
"""

import numpy as np
import pandas as pd


# ============================================================
# Dataset-specific label columns
# ============================================================

LABEL_COLUMNS_BY_DATASET = {
    "UNSW-NB15": [
        "dos",
        "exploits",
        "fuzzers",
        "generic",
        "normal",
        "reconnaissance",
    ],
    "CIC-IDS2017": [
        "BENIGN",
        "DDoS",
        "PortScan",
        "DoS_Hulk",
    ],
}


# ============================================================
# PDT configuration
# ============================================================

PDT_CONFIG = {
    "UNSW-NB15": [
        {
            "labels": ("dos", "exploits"),
            "feature_ranges": [(0, 7)],
        },
        {
            "labels": ("fuzzers", "generic"),
            "feature_ranges": [(8, 15)],
        },
        {
            "labels": ("normal", "reconnaissance"),
            "feature_ranges": [(16, 24)],
        },
    ],

    "CIC-IDS2017": [
        {
            "labels": ("BENIGN", "DoS_Hulk"),
            "feature_ranges": [(0, 9), (35, 44)],
        },
        {
            "labels": ("DDoS", "PortScan"),
            "feature_ranges": [(20, 29), (51, 60)],
        },
    ],
}


# ============================================================
# Targeted label flipping configuration
# ============================================================

LABEL_FLIP_CONFIG = {
    "UNSW-NB15": [
        ("dos", "exploits"),
        ("fuzzers", "generic"),
        ("normal", "reconnaissance"),
    ],

    "CIC-IDS2017": [
        ("BENIGN", "DoS_Hulk"),
        ("DDoS", "PortScan"),
    ],
}


# ============================================================
# Utility functions
# ============================================================

def get_label_columns(dataset_name: str) -> list:
    """
    Return One-Hot label columns for the selected dataset.
    """
    if dataset_name not in LABEL_COLUMNS_BY_DATASET:
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name}. "
            f"Supported datasets: {list(LABEL_COLUMNS_BY_DATASET.keys())}"
        )

    return LABEL_COLUMNS_BY_DATASET[dataset_name]


def get_feature_columns(df: pd.DataFrame, dataset_name: str) -> list:
    """
    Return feature columns by excluding One-Hot label columns.

    This function is kept for compatibility.
    In the current pipeline, X_train and y_train are usually already separated.
    """
    label_columns = get_label_columns(dataset_name)
    return [
        col for col in df.columns
        if col not in label_columns
    ]


def expand_feature_ranges(feature_ranges: list) -> list:
    """
    Convert feature ranges into feature indices.

    Example
    -------
    [(0, 7), (35, 44)] -> [0, 1, ..., 7, 35, 36, ..., 44]
    """
    feature_indices = []

    for start, end in feature_ranges:
        feature_indices.extend(list(range(start, end + 1)))

    return feature_indices


def validate_one_hot_labels(y: pd.DataFrame, label_columns: list) -> None:
    """
    Check whether required One-Hot label columns exist.
    """
    missing_columns = [
        col for col in label_columns
        if col not in y.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing One-Hot label columns: {missing_columns}. "
            f"Existing columns: {list(y.columns)}"
        )


def get_class_indices(y: pd.DataFrame, class_name: str) -> np.ndarray:
    """
    Get row-position indices belonging to a specific One-Hot encoded class.

    The returned indices are safe for numpy indexing even if the DataFrame
    index is not consecutive, because this function returns integer positions.
    """
    if class_name not in y.columns:
        raise ValueError(f"Label column not found: {class_name}")

    mask = y[class_name].to_numpy() == 1
    return np.where(mask)[0]


def validate_attack_strength(
    tamper_ratio: float = None,
    alpha: float = None,
    flip_ratio: float = None,
) -> None:
    """
    Validate attack strength parameters.
    """
    if tamper_ratio is not None:
        if tamper_ratio <= 0.0 or tamper_ratio > 1.0:
            raise ValueError(
                f"tamper_ratio must be in (0, 1], got {tamper_ratio}"
            )

    if alpha is not None:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(
                f"alpha must be in [0, 1], got {alpha}"
            )

    if flip_ratio is not None:
        if flip_ratio <= 0.0 or flip_ratio > 1.0:
            raise ValueError(
                f"flip_ratio must be in (0, 1], got {flip_ratio}"
            )


def get_valid_feature_indices(
    feature_indices: list,
    num_features: int,
    label_a: str,
    label_b: str,
) -> list:
    """
    Remove feature indices that exceed the feature dimension.
    """
    valid_feature_indices = [
        idx for idx in feature_indices
        if 0 <= idx < num_features
    ]

    if len(valid_feature_indices) == 0:
        print(
            f"[PDT warning] Skipped pair ({label_a}, {label_b}) "
            f"because no valid feature indices were found."
        )

    return valid_feature_indices


# ============================================================
# Partial Data Tampering attack
# ============================================================

def tamper_pair_by_opposite_mean_shift(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_a: str,
    label_b: str,
    feature_indices: list,
    tamper_ratio: float = 0.3,
    alpha: float = 0.7,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply clean-label PDT between two classes.

    For selected samples from label_a and label_b, only the specified
    feature indices are moved toward the opposite class feature mean.
    Labels are preserved.
    """
    validate_attack_strength(tamper_ratio=tamper_ratio, alpha=alpha)

    rng = np.random.default_rng(random_state)
    X_poisoned = X.copy()

    idx_a = get_class_indices(y, label_a)
    idx_b = get_class_indices(y, label_b)

    if len(idx_a) == 0 or len(idx_b) == 0:
        print(
            f"[PDT warning] Skipped pair ({label_a}, {label_b}) "
            f"because one of the classes has no samples."
        )
        return X_poisoned

    valid_feature_indices = get_valid_feature_indices(
        feature_indices=feature_indices,
        num_features=X_poisoned.shape[1],
        label_a=label_a,
        label_b=label_b,
    )

    if len(valid_feature_indices) == 0:
        return X_poisoned

    n_a = max(1, int(len(idx_a) * tamper_ratio))
    n_b = max(1, int(len(idx_b) * tamper_ratio))

    n_a = min(n_a, len(idx_a))
    n_b = min(n_b, len(idx_b))

    selected_a = rng.choice(
        idx_a,
        size=n_a,
        replace=False
    )

    selected_b = rng.choice(
        idx_b,
        size=n_b,
        replace=False
    )

    values = X_poisoned.to_numpy(dtype=np.float32)

    mean_a = values[np.ix_(idx_a, valid_feature_indices)].mean(axis=0)
    mean_b = values[np.ix_(idx_b, valid_feature_indices)].mean(axis=0)

    before_a = values[np.ix_(selected_a, valid_feature_indices)].copy()
    before_b = values[np.ix_(selected_b, valid_feature_indices)].copy()

    values[np.ix_(selected_a, valid_feature_indices)] = (
        (1.0 - alpha) * values[np.ix_(selected_a, valid_feature_indices)]
        + alpha * mean_b
    )

    values[np.ix_(selected_b, valid_feature_indices)] = (
        (1.0 - alpha) * values[np.ix_(selected_b, valid_feature_indices)]
        + alpha * mean_a
    )

    after_a = values[np.ix_(selected_a, valid_feature_indices)]
    after_b = values[np.ix_(selected_b, valid_feature_indices)]

    changed_a = int(np.sum(before_a != after_a))
    changed_b = int(np.sum(before_b != after_b))

    X_poisoned.iloc[:, :] = values

    print(
        f"[PDT pair mean_shift] {label_a} <-> {label_b}, "
        f"random_state={random_state}, "
        f"tampered_{label_a}={len(selected_a)}, "
        f"tampered_{label_b}={len(selected_b)}, "
        f"features={valid_feature_indices}, "
        f"changed_values={changed_a + changed_b}, "
        f"tamper_ratio={tamper_ratio}, "
        f"alpha={alpha}"
    )

    return X_poisoned


def tamper_pair_by_feature_swap(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_a: str,
    label_b: str,
    feature_indices: list,
    tamper_ratio: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply clean-label feature swapping between two classes.

    This version directly replaces selected feature values with values
    sampled from the opposite class.
    """
    validate_attack_strength(tamper_ratio=tamper_ratio)

    rng = np.random.default_rng(random_state)
    X_poisoned = X.copy()

    idx_a = get_class_indices(y, label_a)
    idx_b = get_class_indices(y, label_b)

    if len(idx_a) == 0 or len(idx_b) == 0:
        print(
            f"[PDT warning] Skipped pair ({label_a}, {label_b}) "
            f"because one of the classes has no samples."
        )
        return X_poisoned

    valid_feature_indices = get_valid_feature_indices(
        feature_indices=feature_indices,
        num_features=X_poisoned.shape[1],
        label_a=label_a,
        label_b=label_b,
    )

    if len(valid_feature_indices) == 0:
        return X_poisoned

    n_a = max(1, int(len(idx_a) * tamper_ratio))
    n_b = max(1, int(len(idx_b) * tamper_ratio))

    n_a = min(n_a, len(idx_a))
    n_b = min(n_b, len(idx_b))

    selected_a = rng.choice(
        idx_a,
        size=n_a,
        replace=False
    )

    selected_b = rng.choice(
        idx_b,
        size=n_b,
        replace=False
    )

    donor_for_a = rng.choice(
        idx_b,
        size=n_a,
        replace=True
    )

    donor_for_b = rng.choice(
        idx_a,
        size=n_b,
        replace=True
    )

    values = X_poisoned.to_numpy(dtype=np.float32)

    before_a = values[np.ix_(selected_a, valid_feature_indices)].copy()
    before_b = values[np.ix_(selected_b, valid_feature_indices)].copy()

    values[np.ix_(selected_a, valid_feature_indices)] = values[
        np.ix_(donor_for_a, valid_feature_indices)
    ]

    values[np.ix_(selected_b, valid_feature_indices)] = values[
        np.ix_(donor_for_b, valid_feature_indices)
    ]

    after_a = values[np.ix_(selected_a, valid_feature_indices)]
    after_b = values[np.ix_(selected_b, valid_feature_indices)]

    changed_a = int(np.sum(before_a != after_a))
    changed_b = int(np.sum(before_b != after_b))

    X_poisoned.iloc[:, :] = values

    print(
        f"[PDT pair swap] {label_a} <-> {label_b}, "
        f"random_state={random_state}, "
        f"tampered_{label_a}={len(selected_a)}, "
        f"tampered_{label_b}={len(selected_b)}, "
        f"features={valid_feature_indices}, "
        f"changed_values={changed_a + changed_b}, "
        f"tamper_ratio={tamper_ratio}"
    )

    return X_poisoned


def partial_data_tampering(
    X: pd.DataFrame,
    y: pd.DataFrame,
    dataset_name: str,
    tamper_ratio: float = 0.3,
    alpha: float = 0.7,
    random_state: int = 42,
    mode: str = "mean_shift",
) -> pd.DataFrame:
    """
    Apply Table-based Partial Data Tampering attack.

    Modes
    -----
    mean_shift:
        Move selected features toward opposite-class mean.

    swap:
        Replace selected features with opposite-class samples.

    Notes
    -----
    This is a clean-label attack. The feature values are modified,
    but the One-Hot labels are not changed.
    """
    validate_attack_strength(tamper_ratio=tamper_ratio, alpha=alpha)

    if dataset_name not in PDT_CONFIG:
        raise ValueError(
            f"Unsupported dataset_name for PDT: {dataset_name}. "
            f"Supported datasets: {list(PDT_CONFIG.keys())}"
        )

    label_columns = get_label_columns(dataset_name)
    validate_one_hot_labels(y, label_columns)

    if mode not in ["mean_shift", "swap"]:
        raise ValueError(
            f"Unsupported PDT mode: {mode}. "
            "Supported modes are 'mean_shift' and 'swap'."
        )

    X_poisoned = X.copy()
    original_values = X_poisoned.to_numpy(dtype=np.float32).copy()

    print(
        f"[PDT START] dataset={dataset_name}, "
        f"mode={mode}, "
        f"random_state={random_state}, "
        f"tamper_ratio={tamper_ratio}, "
        f"alpha={alpha}"
    )

    for pair_id, config in enumerate(PDT_CONFIG[dataset_name]):
        label_a, label_b = config["labels"]
        feature_indices = expand_feature_ranges(config["feature_ranges"])
        pair_seed = random_state + pair_id

        print(
            f"[PDT] Dataset={dataset_name}, "
            f"Labels=({label_a}, {label_b}), "
            f"Features={feature_indices}, "
            f"Tamper ratio={tamper_ratio}, "
            f"Alpha={alpha}, "
            f"Mode={mode}, "
            f"Pair seed={pair_seed}"
        )

        if mode == "mean_shift":
            X_poisoned = tamper_pair_by_opposite_mean_shift(
                X=X_poisoned,
                y=y,
                label_a=label_a,
                label_b=label_b,
                feature_indices=feature_indices,
                tamper_ratio=tamper_ratio,
                alpha=alpha,
                random_state=pair_seed,
            )

        elif mode == "swap":
            X_poisoned = tamper_pair_by_feature_swap(
                X=X_poisoned,
                y=y,
                label_a=label_a,
                label_b=label_b,
                feature_indices=feature_indices,
                tamper_ratio=tamper_ratio,
                random_state=pair_seed,
            )

    X_poisoned = X_poisoned.clip(lower=0.0, upper=1.0)

    poisoned_values = X_poisoned.to_numpy(dtype=np.float32)
    total_changed_values = int(np.sum(original_values != poisoned_values))

    print(
        f"[PDT SUMMARY] dataset={dataset_name}, "
        f"mode={mode}, "
        f"random_state={random_state}, "
        f"tamper_ratio={tamper_ratio}, "
        f"alpha={alpha}, "
        f"total_changed_feature_values={total_changed_values}, "
        f"labels_changed=0"
    )

    return X_poisoned


# ============================================================
# Label-flipping attack
# ============================================================

def label_flipping(
    y: pd.DataFrame,
    dataset_name: str = None,
    flip_ratio: float = 0.3,
    random_state: int = 42,
    mode: str = "targeted_pair",
) -> pd.DataFrame:
    """
    Apply label-flipping attack to One-Hot encoded labels.

    Modes
    -----
    random:
        Randomly select samples and flip each label to a random wrong class.

    targeted_pair:
        Flip labels between predefined label pairs.
        This mode creates systematic label poisoning and is usually more
        effective for MQV and MPDD analysis.
    """
    validate_attack_strength(flip_ratio=flip_ratio)

    rng = np.random.default_rng(random_state)
    y_poisoned = y.copy()

    print(
        f"[LABEL FLIP START] mode={mode}, "
        f"dataset={dataset_name}, "
        f"random_state={random_state}, "
        f"flip_ratio={flip_ratio}"
    )

    if len(y_poisoned) == 0:
        return y_poisoned

    if mode == "random":
        n_samples = len(y_poisoned)
        n_classes = y_poisoned.shape[1]

        n_flip = max(1, int(n_samples * flip_ratio))
        n_flip = min(n_flip, n_samples)

        flip_indices = rng.choice(
            np.arange(n_samples),
            size=n_flip,
            replace=False
        )

        y_values = y_poisoned.to_numpy(dtype=np.float32)
        original_labels = np.argmax(y_values, axis=1)

        for idx in flip_indices:
            old_label = original_labels[idx]

            candidate_labels = [
                class_id for class_id in range(n_classes)
                if class_id != old_label
            ]

            new_label = rng.choice(candidate_labels)

            y_values[idx, :] = 0.0
            y_values[idx, new_label] = 1.0

        y_poisoned.iloc[:, :] = y_values

        print(
            f"[LABEL FLIP] mode=random, "
            f"random_state={random_state}, "
            f"flipped_samples={n_flip}, "
            f"flip_ratio={flip_ratio}, "
            f"features_changed=0"
        )

        return y_poisoned

    if mode == "targeted_pair":
        if dataset_name is None:
            raise ValueError(
                "dataset_name is required when label_flip mode is 'targeted_pair'."
            )

        if dataset_name not in LABEL_FLIP_CONFIG:
            raise ValueError(
                f"Unsupported dataset_name for targeted label flipping: "
                f"{dataset_name}. Supported datasets: "
                f"{list(LABEL_FLIP_CONFIG.keys())}"
            )

        label_columns = get_label_columns(dataset_name)
        validate_one_hot_labels(y_poisoned, label_columns)

        y_values = y_poisoned.to_numpy(dtype=np.float32)
        total_flipped = 0

        for pair_id, (label_a, label_b) in enumerate(
            LABEL_FLIP_CONFIG[dataset_name]
        ):
            if label_a not in y_poisoned.columns or label_b not in y_poisoned.columns:
                print(
                    f"[LABEL FLIP warning] Skipped pair ({label_a}, {label_b}) "
                    f"because one of the label columns does not exist."
                )
                continue

            idx_a = get_class_indices(y_poisoned, label_a)
            idx_b = get_class_indices(y_poisoned, label_b)

            if len(idx_a) == 0 or len(idx_b) == 0:
                print(
                    f"[LABEL FLIP warning] Skipped pair ({label_a}, {label_b}) "
                    f"because one of the classes has no samples."
                )
                continue

            n_a = max(1, int(len(idx_a) * flip_ratio))
            n_b = max(1, int(len(idx_b) * flip_ratio))

            n_a = min(n_a, len(idx_a))
            n_b = min(n_b, len(idx_b))

            selected_a = rng.choice(
                idx_a,
                size=n_a,
                replace=False
            )

            selected_b = rng.choice(
                idx_b,
                size=n_b,
                replace=False
            )

            col_a = y_poisoned.columns.get_loc(label_a)
            col_b = y_poisoned.columns.get_loc(label_b)

            # label_a -> label_b
            y_values[selected_a, :] = 0.0
            y_values[selected_a, col_b] = 1.0

            # label_b -> label_a
            y_values[selected_b, :] = 0.0
            y_values[selected_b, col_a] = 1.0

            total_flipped += len(selected_a) + len(selected_b)

            print(
                f"[LABEL FLIP pair] {label_a} <-> {label_b}, "
                f"flipped_{label_a}_to_{label_b}={len(selected_a)}, "
                f"flipped_{label_b}_to_{label_a}={len(selected_b)}, "
                f"flip_ratio={flip_ratio}"
            )

        y_poisoned.iloc[:, :] = y_values

        print(
            f"[LABEL FLIP SUMMARY] dataset={dataset_name}, "
            f"mode=targeted_pair, "
            f"random_state={random_state}, "
            f"flip_ratio={flip_ratio}, "
            f"total_flipped_labels={total_flipped}, "
            f"features_changed=0"
        )

        return y_poisoned

    raise ValueError(
        f"Unsupported label_flip mode: {mode}. "
        "Supported modes are 'random' and 'targeted_pair'."
    )


# ============================================================
# Random weight attack
# ============================================================

def random_weight_attack(
    model,
    mode: str = "random",
    noise_scale: float = 1.0,
    random_state: int = 42,
):
    """
    Apply random weight attack to a trained local model before upload.
    """
    rng = np.random.default_rng(random_state)

    weights = model.get_weights()
    poisoned_weights = []

    for weight in weights:
        if mode == "random":
            new_weight = rng.normal(
                loc=0.0,
                scale=noise_scale,
                size=weight.shape
            ).astype(weight.dtype)

        elif mode == "noise":
            noise = rng.normal(
                loc=0.0,
                scale=noise_scale,
                size=weight.shape
            ).astype(weight.dtype)

            new_weight = weight + noise

        else:
            raise ValueError(
                f"Unsupported random weight attack mode: {mode}. "
                "Supported modes are 'random' and 'noise'."
            )

        poisoned_weights.append(new_weight)

    model.set_weights(poisoned_weights)

    print(
        f"[RANDOM WEIGHT ATTACK] mode={mode}, "
        f"noise_scale={noise_scale}, "
        f"random_state={random_state}"
    )

    return model


# ============================================================
# Unified attack interface
# ============================================================

def apply_data_attack(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    dataset_name: str,
    attack_type: str = "none",
    tamper_ratio: float = 0.3,
    alpha: float = 0.7,
    flip_ratio: float = 0.3,
    random_state: int = 42,
    pdt_mode: str = "mean_shift",
    label_flip_mode: str = "targeted_pair",
):
    """
    Apply data-level poisoning attack before local model training.
    """
    print(
        f"[ATTACK START] attack_type={attack_type}, "
        f"dataset={dataset_name}, "
        f"random_state={random_state}"
    )

    if attack_type == "none":
        return X_train, y_train

    if attack_type == "pdt":
        X_train = partial_data_tampering(
            X=X_train,
            y=y_train,
            dataset_name=dataset_name,
            tamper_ratio=tamper_ratio,
            alpha=alpha,
            random_state=random_state,
            mode=pdt_mode,
        )

        return X_train, y_train

    if attack_type == "label_flip":
        y_train = label_flipping(
            y=y_train,
            dataset_name=dataset_name,
            flip_ratio=flip_ratio,
            random_state=random_state,
            mode=label_flip_mode,
        )

        return X_train, y_train

    raise ValueError(
        f"Unknown data attack_type: {attack_type}. "
        "Supported data attack types are 'none', 'pdt', and 'label_flip'."
    )
