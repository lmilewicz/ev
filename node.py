from abc import ABCMeta, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp

from config import Config


class Node(object, metaclass=ABCMeta):
    def __init__(self, params_dict):
            # if 'units' in params_dict: self.units = params_dict['units']
            # else: self.units = 32
            self.units = 32
            if 'activation' in params_dict: self.activation = params_dict['activation']
            if 'kernel_size' in params_dict: self.kernel_size = params_dict['kernel_size']
            if 'dtype' in params_dict: self.dtype = params_dict['dtype']

            self.prob_layer = False
            if 'prob_layer' in params_dict: self.prob_layer = params_dict['prob_layer']

    @abstractmethod
    def create_node(self) -> tf.keras.layers.Layer:
        raise NotImplementedError('Subclass of Node does not implement create_node()')


class DenseLayer(Node):
    def __init__(self, params_dict):
        super().__init__(params_dict)

    def create_node(self) -> tf.keras.layers.Layer:
        if self.prob_layer:
            config = Config()
            return tfp.layers.DenseFlipout(units=self.units,
                                    kernel_divergence_fn=config.kl_divergence_function,
                                    activation=self.activation,
                                    dtype=self.dtype)
        return tf.keras.layers.Dense(units=self.units,
                                    activation=self.activation,
                                    dtype=self.dtype)


class Convolution2D(Node):
    def __init__(self, params_dict):
        super().__init__(params_dict)

    def create_node(self) -> tf.keras.layers.Layer:                                      
        if self.prob_layer:
            config = Config()
            return tfp.layers.Convolution2DFlipout(filters=self.units,
                                    kernel_size=self.kernel_size,
                                    padding='same',
                                    activation=self.activation,
                                    kernel_divergence_fn=config.kl_divergence_function,
                                    dtype=self.dtype)
        return tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=self.kernel_size,
                                    padding='same',
                                    activation=self.activation,
                                    dtype=self.dtype)


def MaxPool2D():
    return tf.keras.layers.MaxPool2D(pool_size=(2, 2))
