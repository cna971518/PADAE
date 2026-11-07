"""Local training with persistent optimizer state and fixed per-client seeds."""
from __future__ import annotations

import random
import numpy as np
from attacks import random_weights, training_data


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    if hasattr(tf.keras.utils, 'set_random_seed'):
        tf.keras.utils.set_random_seed(seed)


def train_client(model, data, client_id, config, seed_function=set_seed):
    seed = config.seed + client_id
    seed_function(seed)
    scenario = config.scenario if client_id in config.malicious_ids else 'clean'
    x, y, events = training_data(data.x, data.y, config.dataset, scenario, seed)
    model.fit(x, y, epochs=config.local_epochs, batch_size=config.batch_size,
              shuffle=False, verbose=config.fit_verbose)
    if scenario == 'rw_1p0':
        model.set_weights(random_weights(model.get_weights(), seed, 1.0))
        events = [{'attack': 'random_weight', 'scale': 1.0,
                   'tensors_replaced': len(model.get_weights())}]
    return events
