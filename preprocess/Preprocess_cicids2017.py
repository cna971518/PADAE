# -*- coding: utf-8 -*-
"""
Preprocessing script for the CIC-IDS2017 dataset.

This script performs:
1. Raw CIC-IDS2017 CSV loading
2. Selection of BENIGN, DDoS, PortScan, and DoS Hulk classes
3. Data cleaning and column renaming
4. Missing and infinite value handling
5. Label One-Hot encoding
6. Train/test split
7. Min-Max normalization
8. Federated client split for 10-client and 20-client settings

Expected project structure:

PADAE/
├── data/
│   ├── raw/
│   │   └── CIC-IDS2017/
│   │       ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   │       ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   │       └── Wednesday-workingHours.pcap_ISCX.csv
│   └── processed/
│       └── CIC-IDS2017/
└── preprocess/
    └── preprocess_cicids2017.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# ============================================================
# Path settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CIC-IDS2017"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "CIC-IDS2017"

CLIENT_SETTINGS = [10, 20]

RANDOM_STATE = 8
TEST_SIZE = 0.3


RAW_FILES = {
    "ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "dos_hulk": "Wednesday-workingHours.pcap_ISCX.csv",
}


COLUMN_NAMES = [
    "Destination_Port", "Flow_Duration", "Total_Fwd_Packets", "Total_Backward_Packets",
    "Total_Length_of_Fwd_Packets", "Total_Length_of_Bwd_Packets", "Fwd_Packet_Length_Max",
    "Fwd_Packet_Length_Min", "Fwd_Packet_Length_Mean", "Fwd_Packet_Length_Std",
    "Bwd_Packet_Length_Max", "Bwd_Packet_Length_Min", "Bwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Std", "Flow_Bytes/s", "Flow_Packets/s", "Flow_IAT_Mean",
    "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", "Fwd_IAT_Total", "Fwd_IAT_Mean",
    "Fwd_IAT_Std", "Fwd_IAT_Max", "Fwd_IAT_Min", "Bwd_IAT_Total", "Bwd_IAT_Mean",
    "Bwd_IAT_Std", "Bwd_IAT_Max", "Bwd_IAT_Min", "Fwd_PSH_Flags", "Bwd_PSH_Flags",
    "Fwd_URG_Flags", "Bwd_URG_Flags", "Fwd_Header_Length", "Bwd_Header_Length",
    "Fwd_Packets/s", "Bwd_Packets/s", "Min_Packet_Length", "Max_Packet_Length",
    "Packet_Length_Mean", "Packet_Length_Std", "Packet_Length_Variance", "FIN_Flag_Count",
    "SYN_Flag_Count", "RST_Flag_Count", "PSH_Flag_Count", "ACK_Flag_Count",
    "URG_Flag_Count", "CWE_Flag_Count", "ECE_Flag_Count", "Down/Up_Ratio",
    "Average_Packet_Size", "Avg_Fwd_Segment_Size", "Avg_Bwd_Segment_Size",
    "Fwd_Header_Length.1", "Fwd_Avg_Bytes/Bulk", "Fwd_Avg_Packets/Bulk",
    "Fwd_Avg_Bulk_Rate", "Bwd_Avg_Bytes/Bulk", "Bwd_Avg_Packets/Bulk",
    "Bwd_Avg_Bulk_Rate", "Subflow_Fwd_Packets", "Subflow_Fwd_Bytes",
    "Subflow_Bwd_Packets", "Subflow_Bwd_Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active_Mean", "Active_Std", "Active_Max", "Active_Min", "Idle_Mean",
    "Idle_Std", "Idle_Max", "Idle_Min", "Label"
]


# ============================================================
# Loading and class selection
# ============================================================

def read_raw_csv(file_key: str) -> pd.DataFrame:
    """Read a raw CIC-IDS2017 CSV file."""
    file_path = RAW_DIR / RAW_FILES[file_key]

    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")

    return pd.read_csv(file_path, low_memory=False)


def load_selected_classes() -> pd.DataFrame:
    """
    Load selected CIC-IDS2017 traffic classes.

    Selected classes:
    - BENIGN
    - DDoS
    - PortScan
    - DoS Hulk
    """
    ddos_df = read_raw_csv("ddos")
    portscan_df = read_raw_csv("portscan")
    dos_hulk_df = read_raw_csv("dos_hulk")

    benign = portscan_df.loc[portscan_df[" Label"] == "BENIGN"]
    ddos = ddos_df.loc[ddos_df[" Label"] == "DDoS"]
    portscan = portscan_df.loc[portscan_df[" Label"] == "PortScan"].head(130000)
    dos_hulk = dos_hulk_df.loc[dos_hulk_df[" Label"] == "DoS Hulk"].head(130000)

    dataframe = pd.concat(
        [benign, ddos, portscan, dos_hulk],
        axis=0,
        ignore_index=True
    )

    dataframe = sklearn.utils.shuffle(
        dataframe,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return dataframe


# ============================================================
# Cleaning and formatting
# ============================================================

def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean CIC-IDS2017 data and assign standardized column names."""
    dataframe = dataframe.copy()

    # Fill missing values in the original Flow Bytes/s column.
    if "Flow Bytes/s" in dataframe.columns:
        dataframe.loc[:, "Flow Bytes/s"] = dataframe.loc[:, "Flow Bytes/s"].replace(
            np.nan, 0, regex=True
        )

    if len(dataframe.columns) != len(COLUMN_NAMES):
        raise ValueError(
            f"Column number mismatch. Expected {len(COLUMN_NAMES)}, "
            f"but got {len(dataframe.columns)}."
        )

    dataframe.columns = COLUMN_NAMES

    # Replace infinite values and missing values in features.
    dataframe = dataframe.replace([np.inf, -np.inf], 0)
    dataframe = dataframe.replace(np.nan, 0)

    # Normalize label names for file compatibility.
    dataframe["Label"] = dataframe["Label"].astype(str).str.strip()
    dataframe["Label"] = dataframe["Label"].replace({
        "BENIGN": "BENIGN",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "DoS Hulk": "DoS_Hulk",
    })

    return dataframe.reset_index(drop=True)


def split_features_labels(dataframe: pd.DataFrame):
    """Separate features and labels."""
    X = dataframe.drop(columns=["Label"])
    y = dataframe["Label"]

    X = X.astype(np.float32)

    return X, y


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


def encode_labels(y_train: pd.Series, y_test: pd.Series):
    """Apply One-Hot encoding to train and test labels using aligned columns."""
    y_train_onehot = pd.get_dummies(y_train)
    y_test_onehot = pd.get_dummies(y_test)

    y_test_onehot = y_test_onehot.reindex(
        columns=y_train_onehot.columns,
        fill_value=0
    )

    return y_train_onehot, y_test_onehot


# ============================================================
# Dataset construction
# ============================================================

def create_train_test_dataset():
    """Create train and test datasets."""
    dataframe = load_selected_classes()
    dataframe = clean_dataset(dataframe)

    X, y = split_features_labels(dataframe)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_test = normalize_train_test(X_train, X_test)
    y_train, y_test = encode_labels(y_train, y_test)

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


def create_federated_datasets(train_df: pd.DataFrame):
    """
    Create federated client datasets for 10-client and 20-client settings.

    For each setting, the training data are divided into:
    - K client datasets
    - 1 server-side validation dataset
    """
    fl_data = sklearn.utils.shuffle(
        train_df,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

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
    print("Start preprocessing CIC-IDS2017 dataset.")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Output directory: {OUT_DIR}")

    train_df, _ = create_train_test_dataset()
    create_federated_datasets(train_df)

    print("CIC-IDS2017 preprocessing completed.")


if __name__ == "__main__":
    main()