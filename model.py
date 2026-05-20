# -*- coding: utf-8 -*-
"""
DNN model construction for PADAE federated learning experiments.
"""

import tensorflow as tf


def get_optimizer(args):
    """
    Build optimizer according to experiment arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Experiment arguments.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        Keras optimizer.
    """
    optimizer_name = getattr(args, "optimizer", "adam").lower()
    learning_rate = getattr(args, "lr", 0.01)

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}. "
        "Supported optimizers are 'adam' and 'sgd'."
    )


def DNN(args, file_name):
    """
    Build a fully connected neural network for multi-class classification.

    The output dimension is automatically determined by the number of
    one-hot label columns defined in args.label_columns.

    Parameters
    ----------
    args : argparse.Namespace
        Experiment arguments.
    file_name : str
        Dataset or client file name associated with this model.

    Returns
    -------
    tf.keras.Model
        Compiled Keras DNN model.
    """
    if args.input_dim is None:
        raise ValueError(
            "args.input_dim is None. Please set input_dim in args.py "
            "or infer it from the preprocessed dataset."
        )

    if not hasattr(args, "label_columns"):
        raise ValueError(
            "args.label_columns is not defined. Please define dataset-specific "
            "one-hot label columns in args.py."
        )

    output_dim = len(args.label_columns)

    model = tf.keras.Sequential(name=f"DNN_{file_name}")

    model.add(
        tf.keras.layers.InputLayer(
            input_shape=(args.input_dim,)
        )
    )

    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))

    model.add(tf.keras.layers.Dense(output_dim, activation="softmax"))

    model.compile(
        optimizer=get_optimizer(args),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.file_name = file_name

    return model