# Plan:
# Create a class for layers;
# Create models based on x layers with BNN;
# Evaluate nn timing on MNIST and CIFAR -> TF vs PyTorch

import tensorflow as tf
import tensorflow_probability as tfp

class DenseLayer():
    def __init__(self):
        self.units = 10
        self.dtype = float
        self.activation = tf.nn.relu

    def create_module_layers(self) -> tf.keras.layers.Layer:
        dense_layer = tf.keras.layers.Dense(units=self.units,
                                    activation=self.activation,
                                    # kernel_initializer=self.kernel_init,
                                    # bias_initializer=self.bias_init,
                                    dtype=self.dtype)
        return dense_layer

class DenseFlipout():
    def __init__(self):
        self.units = 10
        self.dtype = float
        self.activation = tf.nn.relu

    def create_module_layers(self) -> tf.keras.layers.Layer:
        dense_layer = tfp.layers.DenseFlipout(units=self.units,
                                    activation=self.activation,
                                    # kernel_initializer=self.kernel_init,
                                    # bias_initializer=self.bias_init,
                                    dtype=self.dtype)
        return dense_layer

tfp.layers.DenseFlipout(16, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu, name="dense_tfp_1"),


