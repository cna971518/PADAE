# -*- coding: utf-8 -*-
"""
Data loading utilities for PADAE federated learning experiments.

This module loads preprocessed client datasets according to:
- dataset name
- number of clients
- client directory
- dataset-specific one-hot label columns
- dataset-specific local train/test split ratio

Reproducibility design:
- Local train/test split uses args.seed.
- Stratified splitting is applied based on one-hot labels.
- Validation set is loaded deterministically without random shuffling.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    """
    Read a CSV file with common encodings.
    """
    encodings = ["utf-8", "utf-8-sig", "gbk"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Failed to read {file_path} with encodings {encodings}. "
        f"Last error: {last_error}"
    )


def get_test_size(dataset_name: str) -> float:
    """
    Return dataset-specific local train/test split ratio.

    UNSW-NB15 uses a 9:1 train-test split.
    CIC-IDS2017 uses a 7:3 train-test split.
    """
    if dataset_name == "UNSW-NB15":
        return 0.1

    if dataset_name == "CIC-IDS2017":
        return 0.3

    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def validate_label_columns(
    df: pd.DataFrame,
    label_columns: list,
    file_path: Path
) -> None:
    """
    Check whether all required one-hot label columns exist.
    """
    missing_columns = [
        col for col in label_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing label columns in {file_path}: {missing_columns}. "
            f"Existing columns: {list(df.columns)}"
        )


def split_features_and_labels(df: pd.DataFrame, label_columns: list):
    """
    Split dataframe into feature matrix X and one-hot label matrix y.
    """
    validate_label_columns(df, label_columns, Path("input dataframe"))

    y = df[label_columns].copy()
    X = df.drop(columns=label_columns).copy()

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y


def get_stratify_labels(y: pd.DataFrame):
    """
    Convert one-hot labels into class indices for stratified splitting.

    Example:
        [0, 1, 0, 0] -> 1
        [0, 0, 0, 1] -> 3
    """
    if y is None or len(y) == 0:
        return None

    label_array = y.to_numpy()

    if label_array.ndim != 2:
        return None

    class_labels = np.argmax(label_array, axis=1)

    unique_classes, counts = np.unique(class_labels, return_counts=True)

    # train_test_split with stratify requires each class to have at least 2 samples.
    if np.any(counts < 2):
        print(
            "[Warning] Some classes have fewer than 2 samples. "
            "Stratified split is disabled."
        )
        return None

    return class_labels


def trim_to_batch_size(
    X: pd.DataFrame,
    y: pd.DataFrame,
    batch_size: int
):
    """
    Trim samples so that the dataset size is divisible by batch size.

    This keeps the sample order deterministic.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if batch_size <= 0:
        return X, y

    data_len = int(len(X) / batch_size) * batch_size

    if data_len == 0:
        raise ValueError(
            f"Dataset size {len(X)} is smaller than batch size {batch_size}. "
            "Please reduce the batch size."
        )

    X = X.iloc[:data_len].reset_index(drop=True)
    y = y.iloc[:data_len].reset_index(drop=True)

    return X, y


def dataSet(args, file_name: str, batch: int):
    """
    Load a client dataset and split it into local train/test sets.

    The local test split ratio is selected according to args.dataset:
    - UNSW-NB15: test_size = 0.1
    - CIC-IDS2017: test_size = 0.3

    Reproducibility:
    - random_state = args.seed
    - stratify = one-hot label class index
    """
    client_file = Path(args.client_dir) / f"{file_name}.csv"

    if not client_file.exists():
        raise FileNotFoundError(
            f"Client dataset not found: {client_file}. "
            "Please run the preprocessing script first."
        )

    df = read_csv_with_fallback(client_file)
    validate_label_columns(df, args.label_columns, client_file)

    X, y = split_features_and_labels(df, args.label_columns)

    test_size = get_test_size(args.dataset)
    seed = getattr(args, "seed", 42)

    stratify_labels = get_stratify_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    X_train, y_train = trim_to_batch_size(X_train, y_train, batch)
    X_test, y_test = trim_to_batch_size(X_test, y_test, batch)

    return X_train, X_test, y_train, y_test


def validationSet(args, batch: int):
    """
    Load the server-side validation dataset.

    The validation set is not randomly shuffled here, so the order is fixed.
    """
    val_file = Path(args.val_file)

    if not val_file.exists():
        raise FileNotFoundError(
            f"Server-side validation dataset not found: {val_file}. "
            "Please run the preprocessing script first."
        )

    df = read_csv_with_fallback(val_file)
    validate_label_columns(df, args.label_columns, val_file)

    X_val, y_val = split_features_and_labels(df, args.label_columns)
    X_val, y_val = trim_to_batch_size(X_val, y_val, batch)

    return X_val, y_val