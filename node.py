import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABCMeta, abstractmethod

class Node(object, metaclass=ABCMeta):
    def __init__(self, units, activation):
            self.dtype = tf.float32
            self.units = units
            self.activation = activation

    @abstractmethod
    def create_node(self) -> tf.keras.layers.Layer:
        raise NotImplementedError("Subclass of Node does not implement create_node()'")

class NodeTFP(Node):
    def __init__(self, units, activation):
        super().__init__(units, activation)
        self.kl_divergence_function = None

    def create_node(self, dataset_size) -> tf.keras.layers.Layer:
        self.kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(dataset_size, dtype=self.dtype))
        
class DenseLayer(Node):
    def __init__(self, units, activation):
        super().__init__(units, activation)

    def create_node(self, dataset_size) -> tf.keras.layers.Layer:
        return tf.keras.layers.Dense(units=self.units,
                                    activation=self.activation,
                                    # kernel_initializer=self.kernel_init,
                                    # bias_initializer=self.bias_init,
                                    dtype=self.dtype)
class DenseFlipout(NodeTFP):
    def __init__(self, units, activation):
        super().__init__(units, activation)

    def create_node(self, dataset_size) -> tf.keras.layers.Layer:
        super().create_node(dataset_size)
        return tfp.layers.DenseFlipout(units=self.units,
                                    kernel_divergence_fn=self.kl_divergence_function,
                                    activation=self.activation,
                                    # kernel_initializer=self.kernel_init,
                                    # bias_initializer=self.bias_init,
                                    dtype=self.dtype)

class Convolution2D(Node):
    def __init__(self, units, activation):
        super().__init__(units, activation)

    def create_node(self, dataset_size) -> tf.keras.layers.Layer:
        return tf.keras.layers.Conv2D(units=self.units, 
                                        kernel_size=5, 
                                        strides=(1,1),
                                        padding="same", 
                                        dtype=self.dtype)


class Convolution2DFlipout(NodeTFP):
    def __init__(self, units, activation):
        super().__init__(units, activation)

    def create_node(self, dataset_size) -> tf.keras.layers.Layer:
        super().create_node(dataset_size)
        return tfp.layers.Convolution2DFlipout(units=self.units, 
                                        kernel_size=5, 
                                        strides=(1,1), 
                                        padding="same", 
                                        activation=self.activation, 
                                        kernel_divergence_fn=self.kl_divergence_function,
                                        dtype=self.dtype)

class MaxPool2D(Node):
    def __init__(self, units, activation):
        super().__init__(units, activation)

    def create_node(self, dataset_size) -> tf.keras.layers.Layer:
        return tf.keras.layers.MaxPool2D(strides=(4,4), pool_size=(4,4), padding="same")