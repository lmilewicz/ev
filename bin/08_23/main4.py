import tensorflow as tf
import numpy as np

x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
concatted = tf.keras.layers.Concatenate()([x1, x2])
concatted.shape

inputs = tf.keras.Input(shape=10, dtype=tf.float32)

model = tf.keras.Model(inputs, concatted)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

