# -*- coding: utf-8 -*-
"""
Preprocessing script for the UNSW-NB15 dataset.

This script performs:
1. Raw CSV loading and merging
2. Feature type parsing using the official feature description file
3. Missing value handling
4. Categorical feature cleaning
5. One-Hot encoding and PCA dimensionality reduction
6. Low-frequency class removal
7. Class balancing using SMOTE
8. Train/test split
9. Min-Max normalization
10. Federated client split for 10-client and 20-client settings

Expected project structure:

PADAE/
├── data/
│   ├── raw/
│   │   └── UNSW-NB15/
│   │       ├── UNSW-NB15_1.csv
│   │       ├── UNSW-NB15_2.csv
│   │       ├── UNSW-NB15_3.csv
│   │       ├── UNSW-NB15_4.csv
│   │       └── NUSW-NB15_features.csv
│   └── processed/
│       └── UNSW-NB15/
└── preprocess/
    └── preprocess_UNSW_NB15.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# ============================================================
# Path settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "UNSW-NB15"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "UNSW-NB15"

CLIENT_SETTINGS = [10, 20]

RAW_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

FEATURE_FILE = "NUSW-NB15_features.csv"

DROP_ATTACK_CATEGORIES = ["analysis", "backdoor", "shellcode", "worms"]

LIMITED_CLASSES = {
    "normal": 50000,
    "generic": 50000,
}

RANDOM_STATE = 0
TEST_SIZE = 0.1


# ============================================================
# Loading and feature parsing
# ============================================================

def load_raw_csv() -> pd.DataFrame:
    """Load and merge raw UNSW-NB15 CSV files."""
    dataframes = []

    for file_name in RAW_FILES:
        file_path = RAW_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Raw file not found: {file_path}")

        df = pd.read_csv(file_path, header=None, low_memory=False)
        dataframes.append(df)

    dataframe = pd.concat(dataframes, axis=0, ignore_index=True)
    return dataframe


def extract_feature_types():
    """Extract feature type indices from the official feature description file."""
    feature_path = RAW_DIR / FEATURE_FILE
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature description file not found: {feature_path}")

    feature_info = pd.read_csv(
        feature_path,
        encoding="ISO-8859-1",
        header=None
    ).values

    features = feature_info[1:, 1]
    feature_types = np.array([str(item).lower() for item in feature_info[1:, 2]])

    nominal_cols = np.where(feature_types == "nominal")[0]
    integer_cols = np.where(feature_types == "integer")[0]
    binary_cols = np.where(feature_types == "binary")[0]
    float_cols = np.where(feature_types == "float")[0]

    return nominal_cols, integer_cols, binary_cols, float_cols


# ============================================================
# Basic preprocessing
# ============================================================

def process_dataset() -> pd.DataFrame:
    """Clean raw data and assign official feature names."""
    dataframe = load_raw_csv()
    nominal_cols, integer_cols, binary_cols, float_cols = extract_feature_types()

    # Convert numerical columns to float values.
    dataframe[integer_cols] = dataframe[integer_cols].apply(
        pd.to_numeric, errors="coerce"
    ).astype(np.float32)

    dataframe[binary_cols] = dataframe[binary_cols].apply(
        pd.to_numeric, errors="coerce"
    ).astype(np.float32)

    dataframe[float_cols] = dataframe[float_cols].apply(
        pd.to_numeric, errors="coerce"
    ).astype(np.float32)

    # Clean attack category column.
    dataframe.loc[:, 47] = (
        dataframe.loc[:, 47]
        .replace(np.nan, "normal", regex=True)
        .apply(lambda x: str(x).strip().lower())
    )

    dataframe.loc[:, 47] = (
        dataframe.loc[:, 47]
        .replace("backdoors", "backdoor", regex=True)
        .apply(lambda x: str(x).strip().lower())
    )

    # Replace missing numerical values with 0.
    dataframe.loc[:, integer_cols] = dataframe.loc[:, integer_cols].replace(
        np.nan, 0, regex=True
    )
    dataframe.loc[:, binary_cols] = dataframe.loc[:, binary_cols].replace(
        np.nan, 0, regex=True
    )
    dataframe.loc[:, float_cols] = dataframe.loc[:, float_cols].replace(
        np.nan, 0, regex=True
    )

    # Standardize nominal string values.
    dataframe.loc[:, nominal_cols] = dataframe.loc[:, nominal_cols].applymap(
        lambda x: str(x).strip().lower()
    )

    dataframe.columns = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
        "sload", "dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb",
        "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
        "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt",
        "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
        "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
        "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"
    ]

    # Remove IP address fields that are not used in the experimental feature set.
    dataframe = dataframe.drop(columns=["srcip", "dstip"])

    return dataframe


def onehot_with_pca(
    dataframe: pd.DataFrame,
    column_name: str,
    prefix: str,
    variance_ratio: float = 0.98
) -> pd.DataFrame:
    """Apply One-Hot encoding followed by PCA to one categorical column."""
    onehot_data = pd.get_dummies(dataframe[column_name], prefix=prefix)

    pca = PCA(n_components=variance_ratio, random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(onehot_data)

    pca_columns = [
        f"pca_{prefix}_{i + 1}" for i in range(pca_values.shape[1])
    ]

    return pd.DataFrame(pca_values, columns=pca_columns, index=dataframe.index)


def build_feature_matrix() -> pd.DataFrame:
    """Apply One-Hot encoding and PCA to categorical features."""
    dataframe = process_dataset()

    pca_proto = onehot_with_pca(dataframe, "proto", "proto")
    pca_state = onehot_with_pca(dataframe, "state", "state")
    pca_service = onehot_with_pca(dataframe, "service", "service")

    dataframe = dataframe.drop(columns=["proto", "state", "service"])
    dataframe = pd.concat([dataframe, pca_proto, pca_state, pca_service], axis=1)

    return dataframe


# ============================================================
# Dataset construction
# ============================================================

def remove_low_frequency_classes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove low-frequency attack categories not used in the experiments."""
    for category in DROP_ATTACK_CATEGORIES:
        dataframe = dataframe.drop(
            dataframe[dataframe["attack_cat"] == category].index
        )

    return dataframe.reset_index(drop=True)


def limit_selected_classes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Downsample selected high-frequency classes for federated learning data."""
    selected_parts = []

    for category, limit in LIMITED_CLASSES.items():
        category_df = dataframe[dataframe["attack_cat"] == category]
        category_df = sklearn.utils.shuffle(
            category_df,
            random_state=RANDOM_STATE
        ).head(limit)
        selected_parts.append(category_df)

    other_df = dataframe.copy()
    for category in LIMITED_CLASSES.keys():
        other_df = other_df.drop(other_df[other_df["attack_cat"] == category].index)

    dataframe = pd.concat(selected_parts + [other_df], axis=0, ignore_index=True)
    dataframe = sklearn.utils.shuffle(dataframe, random_state=RANDOM_STATE)

    return dataframe.reset_index(drop=True)


def split_features_labels(dataframe: pd.DataFrame):
    """Separate features and attack-category labels."""
    X = dataframe.drop(columns=["attack_cat", "label"])
    X = X.astype(np.float32)
    y = dataframe["attack_cat"]

    return X, y


def apply_smote(X: pd.DataFrame, y: pd.Series):
    """Apply SMOTE to balance attack categories."""
    print("Class distribution before SMOTE:")
    print(y.value_counts(), "\n")

    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)

    print("Class distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts(), "\n")

    return X_resampled, y_resampled


def normalize_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    """Apply Min-Max normalization using training data statistics."""
    scaler = preprocessing.MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled


def create_train_test_dataset():
    """Create train and test datasets for centralized evaluation."""
    dataframe = build_feature_matrix()
    dataframe = remove_low_frequency_classes(dataframe)

    X, y = split_features_labels(dataframe)
    X, y = apply_smote(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_test = normalize_train_test(X_train, X_test)

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)],
        axis=1
    )

    test_df = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)],
        axis=1
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print(f"Saved train dataset to: {OUT_DIR / 'train.csv'}")
    print(f"Saved test dataset to: {OUT_DIR / 'test.csv'}")

    return train_df, test_df


def create_federated_datasets():
    """Create federated client datasets for 10-client and 20-client settings."""
    dataframe = build_feature_matrix()
    dataframe = remove_low_frequency_classes(dataframe)
    dataframe = limit_selected_classes(dataframe)

    X, y = split_features_labels(dataframe)
    X, y = apply_smote(X, y)

    X = pd.DataFrame(X, columns=X.columns)
    y = pd.Series(y, name="attack_cat")

    X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    y_onehot = pd.get_dummies(y)

    fl_data = pd.concat(
        [X_scaled.reset_index(drop=True), y_onehot.reset_index(drop=True)],
        axis=1
    )

    fl_data = sklearn.utils.shuffle(fl_data, random_state=RANDOM_STATE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fl_data.to_csv(OUT_DIR / "federated_full.csv", index=False)

    for client_count in CLIENT_SETTINGS:
        output_dir = OUT_DIR / f"clients_{client_count}"
        output_dir.mkdir(parents=True, exist_ok=True)

        split_count = client_count + 1
        df_len = len(fl_data) // split_count

        for i in range(client_count):
            start = i * df_len
            end = (i + 1) * df_len
            client_df = fl_data.iloc[start:end].reset_index(drop=True)

            client_file = output_dir / f"client_{i + 1:02d}.csv"
            client_df.to_csv(client_file, index=False)

        val_start = client_count * df_len
        val_end = (client_count + 1) * df_len
        val_df = fl_data.iloc[val_start:val_end].reset_index(drop=True)
        val_df.to_csv(output_dir / "Val.csv", index=False)

        print(f"Saved {client_count} client datasets to: {output_dir}")
        print(f"Saved server-side validation dataset to: {output_dir / 'Val.csv'}")


def main():
    """Run all preprocessing steps."""
    print("Start preprocessing UNSW-NB15 dataset.")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Output directory: {OUT_DIR}")

    create_train_test_dataset()
    create_federated_datasets()

    print("UNSW-NB15 preprocessing completed.")


if __name__ == "__main__":
    main()