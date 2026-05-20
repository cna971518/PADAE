# -*- coding: utf-8 -*-
"""
PADAE federated learning client.

This module handles local client training and local model testing.
If a client is specified as malicious, the selected poisoning attack
will be applied during local training.

Reproducibility design:
- Each client uses args.seed + client_id as its local seed.
- Keras fit uses shuffle=False to avoid random reshuffling per epoch.
- Poisoning attack sampling also uses args.seed + client_id.
"""

import os
import random

import numpy as np
import tensorflow as tf

from data_process import dataSet
from attacks import apply_data_attack, random_weight_attack


def set_client_seed(seed: int):
    """
    Fix random seeds for a single client training process.

    This helps make local model initialization, local training,
    and attack sampling more reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def train(args, nn, file_name, num):
    """
    Train a local client model.

    Parameters
    ----------
    args : argparse.Namespace
        Experiment arguments.
    nn : tf.keras.Model
        Local client model.
    file_name : str
        Client dataset file name without extension.
    num : int
        Client index.

    Returns
    -------
    tf.keras.Model
        Trained local client model.
    """
    client_seed = getattr(args, "seed", 42) + num
    set_client_seed(client_seed)

    print(f"Client {num + 1} training:")
    print(f"[Client Seed] client_{num + 1:02d} seed = {client_seed}")

    X_train, X_test, y_train, y_test = dataSet(args, file_name, args.B)

    is_malicious = num in args.malicious_clients

    if is_malicious:
        print(
            f"Client {num + 1} is malicious. "
            f"Attack type: {args.attack_type}"
        )

        if args.attack_type in ["pdt", "label_flip"]:
            X_before_attack = X_train.copy()
            y_before_attack = y_train.copy()

            X_train, y_train = apply_data_attack(
                X_train=X_train,
                y_train=y_train,
                dataset_name=args.dataset,
                attack_type=args.attack_type,
                tamper_ratio=args.tamper_ratio,
                alpha=args.alpha,
                flip_ratio=args.flip_ratio,
                random_state=client_seed,
                pdt_mode=args.pdt_mode,
                label_flip_mode=args.label_flip_mode,
            )

            changed_X_values = (
                X_before_attack.to_numpy() != X_train.to_numpy()
            ).sum()

            changed_y_values = (
                y_before_attack.to_numpy() != y_train.to_numpy()
            ).sum()

            print(
                f"[ATTACK CHECK] client_{num + 1:02d}, "
                f"attack_type={args.attack_type}, "
                f"changed_X_values={changed_X_values}, "
                f"changed_y_values={changed_y_values}"
            )

            if args.attack_type == "pdt":
                if changed_X_values > 0 and changed_y_values == 0:
                    print(
                        f"[PDT CHECK] client_{num + 1:02d}: "
                        f"PDT was applied correctly. "
                        f"Features changed, labels unchanged."
                    )
                else:
                    print(
                        f"[PDT WARNING] client_{num + 1:02d}: "
                        f"PDT may not have been applied correctly. "
                        f"Expected changed_X_values > 0 and changed_y_values = 0."
                    )

            if args.attack_type == "label_flip":
                if changed_X_values == 0 and changed_y_values > 0:
                    print(
                        f"[LABEL FLIP CHECK] client_{num + 1:02d}: "
                        f"Label flipping was applied correctly. "
                        f"Features unchanged, labels changed."
                    )
                else:
                    print(
                        f"[LABEL FLIP WARNING] client_{num + 1:02d}: "
                        f"Label flipping may not have been applied correctly. "
                        f"Expected changed_X_values = 0 and changed_y_values > 0."
                    )

    else:
        print(f"Client {num + 1} is benign. No data attack applied.")

    nn.len = len(X_train)

    batch_size = args.B
    epochs = args.E

    nn.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=False
    )

    if is_malicious and args.attack_type == "random_weight":
        print(f"Applying random weight attack to Client {num + 1}")

        set_client_seed(client_seed)

        nn = random_weight_attack(
            model=nn,
            mode=args.weight_attack_mode,
            noise_scale=args.weight_noise_scale,
            random_state=client_seed,
        )

    return nn


def test(args, nn):
    """
    Test a local or global model on the corresponding client dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Experiment arguments.
    nn : tf.keras.Model
        Model to be evaluated.

    Returns
    -------
    float
        Test accuracy.
    """
    X_train, X_test, y_train, y_test = dataSet(args, nn.file_name, args.B)

    loss, acc = nn.evaluate(
        X_test,
        y_test,
        batch_size=args.B,
        verbose=0
    )

    print("\n       Test accuracy: %.3f%%" % (100.0 * acc))

    return acc