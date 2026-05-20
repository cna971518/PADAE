# -*- coding: utf-8 -*-
"""
Main entry point for PADAE federated learning experiments.

This script:
1. Parses experiment arguments.
2. Fixes random seeds for reproducibility.
3. Creates a result directory based on experiment settings.
4. Records console logs.
5. Runs federated training.
6. Runs global model testing on all clients.
7. Saves client-level status results.
8. Saves experiment settings and summary results.

Note:
- The experiment folder name is intentionally shortened to avoid
  Windows path-too-long errors.
"""

import csv
import sys
import os
import random
import traceback
from datetime import datetime
from pathlib import Path

from args import args_parser


def set_global_seed(seed=42):
    """
    Fix random seeds for reproducible experiments.

    Notes:
    - This function should be called before importing server / model modules,
      because those modules may import TensorFlow.
    - Deterministic behavior is improved but may still vary slightly depending
      on GPU, CUDA, cuDNN, and TensorFlow versions.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)

    import numpy as np
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

    print(f"[Seed Fixed] Global seed = {seed}")


class TeeLogger:
    """
    Write terminal output to both console and log file.
    """

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def sanitize_for_path(text: str) -> str:
    """
    Convert text into a safe folder-name string.
    """
    text = str(text)

    invalid_chars = [
        "\\", "/", ":", "*", "?", '"', "<", ">", "|",
        " ", "[", "]", "(", ")", "{", "}", ","
    ]

    for ch in invalid_chars:
        text = text.replace(ch, "-")

    while "--" in text:
        text = text.replace("--", "-")

    return text.strip("-")


def compact_float(value):
    """
    Convert float-like values into short path-friendly strings.

    Examples
    --------
    0.1  -> 0p1
    1.0  -> 1
    None -> NA
    """
    if value is None:
        return "NA"

    try:
        value = float(value)

        if value.is_integer():
            text = str(int(value))
        else:
            text = f"{value:g}"

    except Exception:
        text = str(value)

    return text.replace(".", "p")


def build_experiment_name(args) -> str:
    """
    Build a short result folder name based on experiment settings.

    This version avoids Windows path-too-long errors.
    Detailed experiment settings are still saved in experiment_settings.txt
    and summary_results.csv.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_tag = (
        str(args.dataset)
        .replace("CIC-IDS2017", "CIC")
        .replace("UNSW-NB15", "UNSW")
    )

    base_name = (
        f"{dataset_tag}"
        f"_K{args.K}"
        f"_A-{args.attack_type}"
    )

    if args.attack_type != "none":
        if len(args.malicious_clients) > 0:
            malicious_clients = "-".join(
                str(i) for i in args.malicious_clients
            )
        else:
            malicious_clients = "none"

        base_name += f"_M-{malicious_clients}"

        if args.attack_type == "pdt":
            base_name += (
                f"_tr{compact_float(getattr(args, 'tamper_ratio', None))}"
                f"_a{compact_float(getattr(args, 'alpha', None))}"
                f"_{getattr(args, 'pdt_mode', None)}"
            )

        elif args.attack_type == "label_flip":
            base_name += (
                f"_fr{compact_float(getattr(args, 'flip_ratio', None))}"
                f"_{getattr(args, 'label_flip_mode', None)}"
            )

        elif args.attack_type == "random_weight":
            base_name += (
                f"_{getattr(args, 'weight_attack_mode', None)}"
                f"_ns{compact_float(getattr(args, 'weight_noise_scale', None))}"
            )

    base_name += (
        f"_agg-{getattr(args, 'aggregation_method', 'cma')}"
        f"_s{getattr(args, 'seed', 42)}"
        f"_{timestamp}"
    )

    return sanitize_for_path(base_name)


def create_result_dir(args):
    """
    Create a result directory based on experiment settings.
    """
    experiment_name = build_experiment_name(args)

    result_dir = (
        Path("results")
        / args.dataset
        / f"clients_{args.K}"
        / experiment_name
    )

    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def print_experiment_settings(args, result_dir):
    """
    Print key experiment settings.
    """
    print("=" * 60)
    print("PADAE Federated Learning Experiment")
    print("=" * 60)
    print(f"Dataset              : {args.dataset}")
    print(f"Number of clients    : {args.K}")
    print(f"Client directory     : {args.client_dir}")
    print(f"Validation file      : {args.val_file}")
    print(f"Input dimension      : {args.input_dim}")
    print(f"Output classes       : {len(args.label_columns)}")
    print(f"Label columns        : {args.label_columns}")
    print(f"Local epochs         : {args.E}")
    print(f"Global rounds        : {args.r}")
    print(f"Batch size           : {args.B}")
    print(f"Learning rate        : {args.lr}")
    print(f"Optimizer            : {args.optimizer}")
    print(f"Attack type          : {args.attack_type}")
    print(f"Malicious clients    : {args.malicious_clients}")
    print(f"Tamper ratio         : {getattr(args, 'tamper_ratio', None)}")
    print(f"Alpha                : {getattr(args, 'alpha', None)}")
    print(f"PDT mode             : {getattr(args, 'pdt_mode', None)}")
    print(f"Flip ratio           : {getattr(args, 'flip_ratio', None)}")
    print(f"Label flip mode      : {getattr(args, 'label_flip_mode', None)}")
    print(f"Weight attack mode   : {getattr(args, 'weight_attack_mode', None)}")
    print(f"Weight noise scale   : {getattr(args, 'weight_noise_scale', None)}")
    print(f"Aggregation method   : {getattr(args, 'aggregation_method', 'cma')}")
    print(f"CMA beta             : {getattr(args, 'cma_beta', 0.10)}")
    print(f"CMA lambda           : {getattr(args, 'cma_lambda', 0.80)}")
    print(f"KS threshold         : {getattr(args, 'ks_threshold', 0.5)}")
    print(f"P-value threshold    : {getattr(args, 'pvalue_threshold', 0.05)}")
    print(f"Abnormal threshold   : {getattr(args, 'abnormal_round_threshold', None)}")
    print(f"Seed                 : {getattr(args, 'seed', 42)}")
    print(f"Result directory     : {result_dir}")
    print("=" * 60)


def save_experiment_settings(args, result_dir):
    """
    Save experiment settings to a text file.
    """
    setting_file = result_dir / "experiment_settings.txt"
    setting_file.parent.mkdir(parents=True, exist_ok=True)

    with open(setting_file, "w", encoding="utf-8") as f:
        f.write("PADAE Federated Learning Experiment\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset              : {args.dataset}\n")
        f.write(f"Number of clients    : {args.K}\n")
        f.write(f"Client directory     : {args.client_dir}\n")
        f.write(f"Validation file      : {args.val_file}\n")
        f.write(f"Input dimension      : {args.input_dim}\n")
        f.write(f"Output classes       : {len(args.label_columns)}\n")
        f.write(f"Label columns        : {args.label_columns}\n")
        f.write(f"Local epochs         : {args.E}\n")
        f.write(f"Global rounds        : {args.r}\n")
        f.write(f"Batch size           : {args.B}\n")
        f.write(f"Learning rate        : {args.lr}\n")
        f.write(f"Optimizer            : {args.optimizer}\n")
        f.write(f"Attack type          : {args.attack_type}\n")
        f.write(f"Malicious clients    : {args.malicious_clients}\n")
        f.write(f"Tamper ratio         : {getattr(args, 'tamper_ratio', None)}\n")
        f.write(f"Alpha                : {getattr(args, 'alpha', None)}\n")
        f.write(f"PDT mode             : {getattr(args, 'pdt_mode', None)}\n")
        f.write(f"Flip ratio           : {getattr(args, 'flip_ratio', None)}\n")
        f.write(f"Label flip mode      : {getattr(args, 'label_flip_mode', None)}\n")
        f.write(f"Weight attack mode   : {getattr(args, 'weight_attack_mode', None)}\n")
        f.write(f"Weight noise scale   : {getattr(args, 'weight_noise_scale', None)}\n")
        f.write(f"Aggregation method   : {getattr(args, 'aggregation_method', 'cma')}\n")
        f.write(f"CMA beta             : {getattr(args, 'cma_beta', 0.10)}\n")
        f.write(f"CMA lambda           : {getattr(args, 'cma_lambda', 0.80)}\n")
        f.write(f"KS threshold         : {getattr(args, 'ks_threshold', 0.5)}\n")
        f.write(f"P-value threshold    : {getattr(args, 'pvalue_threshold', 0.05)}\n")
        f.write(f"Abnormal threshold   : {getattr(args, 'abnormal_round_threshold', None)}\n")
        f.write(f"Seed                 : {getattr(args, 'seed', 42)}\n")
        f.write(f"Result directory     : {result_dir}\n")


def get_retained_removed_clients(args, fedavg):
    """
    Get retained and removed client names.

    retained_clients:
        clients in the final secure aggregation set.

    removed_clients:
        clients not in the final secure aggregation set.
        This includes warning or permanently removed clients.
    """
    retained_clients = [
        args.clients[i] for i in fedavg.secure_server
    ]

    removed_clients = [
        args.clients[i]
        for i in range(args.K)
        if i not in fedavg.secure_server
    ]

    return retained_clients, removed_clients


def save_summary_results(args, fedavg, final_accuracy, result_dir, status="completed"):
    """
    Save final experiment summary to CSV.
    """
    summary_file = result_dir / "summary_results.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    retained_clients, removed_clients = get_retained_removed_clients(args, fedavg)

    fieldnames = [
        "status",
        "dataset",
        "num_clients",
        "attack_type",
        "malicious_clients",
        "tamper_ratio",
        "alpha",
        "pdt_mode",
        "flip_ratio",
        "label_flip_mode",
        "weight_attack_mode",
        "weight_noise_scale",
        "aggregation_method",
        "cma_beta",
        "cma_lambda",
        "local_epochs",
        "global_rounds",
        "batch_size",
        "learning_rate",
        "optimizer",
        "input_dim",
        "output_classes",
        "ks_threshold",
        "pvalue_threshold",
        "abnormal_round_threshold",
        "seed",
        "retained_client_count",
        "removed_client_count",
        "retained_clients",
        "removed_clients",
        "final_accuracy_all_clients",
        "result_dir",
    ]

    row = {
        "status": status,
        "dataset": args.dataset,
        "num_clients": args.K,
        "attack_type": args.attack_type,
        "malicious_clients": str(args.malicious_clients),
        "tamper_ratio": getattr(args, "tamper_ratio", None),
        "alpha": getattr(args, "alpha", None),
        "pdt_mode": getattr(args, "pdt_mode", None),
        "flip_ratio": getattr(args, "flip_ratio", None),
        "label_flip_mode": getattr(args, "label_flip_mode", None),
        "weight_attack_mode": getattr(args, "weight_attack_mode", None),
        "weight_noise_scale": getattr(args, "weight_noise_scale", None),
        "aggregation_method": getattr(args, "aggregation_method", "cma"),
        "cma_beta": getattr(args, "cma_beta", 0.10),
        "cma_lambda": getattr(args, "cma_lambda", 0.80),
        "local_epochs": args.E,
        "global_rounds": args.r,
        "batch_size": args.B,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "input_dim": args.input_dim,
        "output_classes": len(args.label_columns),
        "ks_threshold": getattr(args, "ks_threshold", 0.5),
        "pvalue_threshold": getattr(args, "pvalue_threshold", 0.05),
        "abnormal_round_threshold": getattr(args, "abnormal_round_threshold", None),
        "seed": getattr(args, "seed", 42),
        "retained_client_count": len(retained_clients),
        "removed_client_count": len(removed_clients),
        "retained_clients": ";".join(retained_clients),
        "removed_clients": ";".join(removed_clients),
        "final_accuracy_all_clients": final_accuracy,
        "result_dir": str(result_dir),
    }

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def append_global_summary(args, fedavg, final_accuracy, result_dir, status="completed"):
    """
    Append final result to a global summary CSV.
    """
    global_summary_file = Path("results") / "all_experiments_summary.csv"
    global_summary_file.parent.mkdir(parents=True, exist_ok=True)

    retained_clients, removed_clients = get_retained_removed_clients(args, fedavg)

    fieldnames = [
        "time",
        "status",
        "dataset",
        "num_clients",
        "attack_type",
        "malicious_clients",
        "tamper_ratio",
        "alpha",
        "pdt_mode",
        "flip_ratio",
        "label_flip_mode",
        "weight_attack_mode",
        "weight_noise_scale",
        "aggregation_method",
        "cma_beta",
        "cma_lambda",
        "local_epochs",
        "global_rounds",
        "batch_size",
        "learning_rate",
        "optimizer",
        "input_dim",
        "output_classes",
        "ks_threshold",
        "pvalue_threshold",
        "abnormal_round_threshold",
        "seed",
        "retained_client_count",
        "removed_client_count",
        "retained_clients",
        "removed_clients",
        "final_accuracy_all_clients",
        "result_dir",
    ]

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "dataset": args.dataset,
        "num_clients": args.K,
        "attack_type": args.attack_type,
        "malicious_clients": str(args.malicious_clients),
        "tamper_ratio": getattr(args, "tamper_ratio", None),
        "alpha": getattr(args, "alpha", None),
        "pdt_mode": getattr(args, "pdt_mode", None),
        "flip_ratio": getattr(args, "flip_ratio", None),
        "label_flip_mode": getattr(args, "label_flip_mode", None),
        "weight_attack_mode": getattr(args, "weight_attack_mode", None),
        "weight_noise_scale": getattr(args, "weight_noise_scale", None),
        "aggregation_method": getattr(args, "aggregation_method", "cma"),
        "cma_beta": getattr(args, "cma_beta", 0.10),
        "cma_lambda": getattr(args, "cma_lambda", 0.80),
        "local_epochs": args.E,
        "global_rounds": args.r,
        "batch_size": args.B,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "input_dim": args.input_dim,
        "output_classes": len(args.label_columns),
        "ks_threshold": getattr(args, "ks_threshold", 0.5),
        "pvalue_threshold": getattr(args, "pvalue_threshold", 0.05),
        "abnormal_round_threshold": getattr(args, "abnormal_round_threshold", None),
        "seed": getattr(args, "seed", 42),
        "retained_client_count": len(retained_clients),
        "removed_client_count": len(removed_clients),
        "retained_clients": ";".join(retained_clients),
        "removed_clients": ";".join(removed_clients),
        "final_accuracy_all_clients": final_accuracy,
        "result_dir": str(result_dir),
    }

    file_exists = global_summary_file.exists()

    with open(global_summary_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_error_log(error, result_dir):
    """
    Save error traceback if the experiment fails.
    """
    error_file = result_dir / "error_log.txt"
    error_file.parent.mkdir(parents=True, exist_ok=True)

    with open(error_file, "w", encoding="utf-8") as f:
        f.write("Experiment failed.\n")
        f.write("=" * 60 + "\n")
        f.write(str(error) + "\n\n")
        f.write(traceback.format_exc())


def main():
    """
    Run PADAE federated learning training and evaluation.
    """
    args = args_parser()

    # Fix experiment seed before importing server / model modules.
    set_global_seed(getattr(args, "seed", 42))

    # Import FedAvg after fixing seed and deterministic TensorFlow settings.
    from server import FedAvg

    result_dir = create_result_dir(args)
    console_log_file = result_dir / "console_log.txt"

    logger = TeeLogger(console_log_file)
    original_stdout = sys.stdout
    sys.stdout = logger

    fedavg = None
    final_accuracy = None

    try:
        print_experiment_settings(args, result_dir)
        save_experiment_settings(args, result_dir)

        fedavg = FedAvg(args)

        print("\nStart federated training:")
        fedavg.server()

        print("\nStart global model testing:")
        final_accuracy = fedavg.global_test()

        # Save every client's ACC, Val_ACC, KS_value, PVA_mean, and aggregation status.
        fedavg.save_client_status_csv(result_dir)

        save_summary_results(
            args=args,
            fedavg=fedavg,
            final_accuracy=final_accuracy,
            result_dir=result_dir,
            status="completed"
        )

        append_global_summary(
            args=args,
            fedavg=fedavg,
            final_accuracy=final_accuracy,
            result_dir=result_dir,
            status="completed"
        )

        print("\nExperiment completed.")
        print(f"Result directory: {result_dir}")
        print(f"Final accuracy on all clients: {final_accuracy:.6f}")

    except Exception as error:
        print("\nExperiment failed.")
        print(f"Error: {error}")

        save_error_log(error, result_dir)

        if fedavg is not None:
            try:
                fedavg.save_client_status_csv(result_dir)
            except Exception:
                pass

            save_summary_results(
                args=args,
                fedavg=fedavg,
                final_accuracy=final_accuracy,
                result_dir=result_dir,
                status="failed"
            )

            append_global_summary(
                args=args,
                fedavg=fedavg,
                final_accuracy=final_accuracy,
                result_dir=result_dir,
                status="failed"
            )

        raise

    finally:
        sys.stdout = original_stdout
        logger.close()


if __name__ == "__main__":
    main()
