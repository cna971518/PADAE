"""The manuscript's five-hidden-layer fully connected classifier."""
from __future__ import annotations


def build_model(input_dim, output_dim, learning_rate=0.01, name='global'):
    import tensorflow as tf
    if input_dim < 1 or output_dim < 2:
        raise ValueError('Invalid model dimensions.')
    model = tf.keras.Sequential(name=f'DNN_{name}')
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    for units in (128, 128, 64, 32, 16):
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
