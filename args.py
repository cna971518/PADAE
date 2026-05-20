# -*- coding: utf-8 -*-
"""
Argument parser for PADAE federated learning experiments.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch


def parse_malicious_clients(client_string: str):
    """
    Parse malicious client indices from a comma-separated string.

    Example
    -------
    "0,1,2,3" -> [0, 1, 2, 3]
    """
    if client_string is None or client_string.strip() == "":
        return []

    return [int(x.strip()) for x in client_string.split(",") if x.strip() != ""]


def infer_input_dim_from_client_file(client_dir: Path, label_columns: list) -> int:
    """
    Infer input feature dimension from client_01.csv.

    The input dimension is calculated as:
    total columns - one-hot label columns
    """
    sample_file = client_dir / "client_01.csv"

    if not sample_file.exists():
        raise FileNotFoundError(
            f"Cannot infer input_dim because sample client file was not found: {sample_file}. "
            "Please run the preprocessing script first."
        )

    df = pd.read_csv(sample_file, low_memory=False)

    missing_labels = [col for col in label_columns if col not in df.columns]

    if missing_labels:
        raise ValueError(
            f"Missing label columns in {sample_file}: {missing_labels}. "
            f"Existing columns: {list(df.columns)}"
        )

    feature_columns = [
        col for col in df.columns
        if col not in label_columns
    ]

    return len(feature_columns)


def args_parser():
    parser = argparse.ArgumentParser(
        description="PADAE federated learning experiment settings"
    )

    # ============================================================
    # Dataset settings
    # ============================================================
    parser.add_argument(
        "--dataset",
        type=str,
        default="UNSW-NB15",
        choices=["UNSW-NB15", "CIC-IDS2017"],
        help="Dataset name."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/processed",
        help="Root directory of processed datasets."
    )

    # ============================================================
    # Federated learning settings
    # ============================================================
    parser.add_argument(
        "--E",
        type=int,
        default=40,
        help="Number of local training epochs."
    )

    parser.add_argument(
        "--r",
        type=int,
        default=12,
        help="Number of global communication rounds."
    )

    parser.add_argument(
        "--K",
        type=int,
        default=10,
        choices=[10, 20],
        help="Number of federated clients."
    )

    parser.add_argument(
        "--input_dim",
        type=int,
        default=None,
        help="Input feature dimension. If not specified, it will be inferred from client_01.csv."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate."
    )

    parser.add_argument(
        "--C",
        type=float,
        default=0.5,
        help="Client sampling rate."
    )

    parser.add_argument(
        "--B",
        type=int,
        default=500,
        help="Local batch size."
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay."
    )

    parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="Computation device."
    )

    parser.add_argument(
    "--aggregation_method",
    type=str,
    default="cma",
    choices=["fedavg", "cma"],
    help="Aggregation method: fedavg or cma."
    )

    parser.add_argument(
    "--cma_beta",
    type=float,
    default=0.10,
    help="Proportion of high-contribution clients in CMA."
    )

    parser.add_argument(
    "--cma_lambda",
    type=float,
    default=0.80,
    help="Total aggregation weight assigned to high-contribution clients in CMA."
    )

    # ============================================================
    # PADAE defense thresholds
    # ============================================================
    parser.add_argument(
        "--ks_threshold",
        type=float,
        default=0.5,
        help="K-S statistic threshold for Model Quality Validation."
    )

    parser.add_argument(
        "--pvalue_threshold",
        type=float,
        default=0.05,
        help="Average p-value threshold for Model Parameter Distribution Detection."
    )
    parser.add_argument(
    "--abnormal_round_threshold",
    type=int,
    default=2,
    help="Number of consecutive abnormal rounds required before removing a client."
    )

    # ============================================================
    # Poisoning attack settings
    # ============================================================
    parser.add_argument(
        "--attack_type",
        type=str,
        default="none",
        choices=["none", "pdt", "label_flip", "random_weight"],
        help="Type of poisoning attack."
    )

    parser.add_argument(
        "--malicious_clients",
        type=str,
        default="",
        help="Comma-separated malicious client indices, e.g., 0,1,2,3."
    )

    parser.add_argument(
        "--tamper_ratio",
        type=float,
        default=1.0,
        help="Ratio of samples tampered in the PDT attack."
    )

    parser.add_argument(
    "--label_flip_mode",
    type=str,
    default="targeted_pair",
    choices=["random", "targeted_pair"],
    help="Label flipping mode: random or targeted_pair."
    )

    parser.add_argument(
        "--flip_ratio",
        type=float,
        default=1.0,
        help="Ratio of labels flipped in the label-flipping attack."
    )

    parser.add_argument(
        "--weight_attack_mode",
        type=str,
        default="random",
        choices=["random", "noise"],
        help="Mode of random weight attack."
    )

    parser.add_argument(
        "--weight_noise_scale",
        type=float,
        default=1.0,
        help="Noise scale for the random weight attack."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    parser.add_argument(
    "--alpha",
    type=float,
    default=0.7,
    help="PDT tampering strength. Larger alpha moves features closer to the opposite class distribution."
    )
    parser.add_argument(
    "--pdt_mode",
    type=str,
    default="mean_shift",
    choices=["mean_shift", "swap"],
    help="PDT mode: mean_shift or swap."
    )



    args = parser.parse_args()

    # ============================================================
    # Derived path settings
    # ============================================================
    project_root = Path(__file__).resolve().parent

    args.project_root = project_root
    args.data_root = Path(args.data_root)

    args.dataset_dir = args.data_root / args.dataset
    args.client_dir = args.dataset_dir / f"clients_{args.K}"
    args.val_file = args.client_dir / "Val.csv"

    # These files are kept for future centralized baseline or global testing.
    # They are not required for the current client-based FL workflow.
    args.train_file = args.dataset_dir / "train.csv"
    args.test_file = args.dataset_dir / "test.csv"

    args.clients = [f"client_{i:02d}" for i in range(1, args.K + 1)]

    # ============================================================
    # Dataset-specific label columns
    # ============================================================
    if args.dataset == "UNSW-NB15":
        args.label_columns = [
            "dos",
            "exploits",
            "fuzzers",
            "generic",
            "normal",
            "reconnaissance",
        ]

    elif args.dataset == "CIC-IDS2017":
        args.label_columns = [
            "BENIGN",
            "DDoS",
            "PortScan",
            "DoS_Hulk",
        ]

    # ============================================================
    # Basic file checks before input_dim inference
    # ============================================================
    if not args.client_dir.exists():
        raise FileNotFoundError(
            f"Client directory not found: {args.client_dir}. "
            "Please run the preprocessing script first."
        )

    if not args.val_file.exists():
        raise FileNotFoundError(
            f"Server-side validation file not found: {args.val_file}. "
            "Please run the preprocessing script first."
        )

    # ============================================================
    # Infer input dimension automatically
    # ============================================================
    if args.input_dim is None:
        args.input_dim = infer_input_dim_from_client_file(
            client_dir=args.client_dir,
            label_columns=args.label_columns
        )

    # ============================================================
    # Parse malicious clients
    # ============================================================
    args.malicious_clients = parse_malicious_clients(args.malicious_clients)

    for client_id in args.malicious_clients:
        if client_id < 0 or client_id >= args.K:
            raise ValueError(
                f"Invalid malicious client index: {client_id}. "
                f"Valid range is 0 to {args.K - 1}."
            )

    return args