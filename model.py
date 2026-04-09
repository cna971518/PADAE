"""
    FedKS-BC 构建模型
"""
import tensorflow as tf

# from tf.keras.models import Sequential
# from tf.keras.layers import Dense, Activation


def DNN(args, file_name):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=args.input_dim))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(6))
    model.add(tf.keras.layers.Activation('softmax'))  # 多分类

    # model.summary()  # 模型各层的参数状况
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
