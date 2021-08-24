import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
from abc import ABCMeta, abstractmethod

class Node(object, metaclass=ABCMeta):
    def __init__(self, units, activation):
            self.units = units
            self.dtype = tf.float32
            self.activation = activation

    @abstractmethod
    def create_node(self) -> tf.keras.layers.Layer:
        raise NotImplementedError("Subclass of Node does not implement create_node()'")

        
class DenseLayer(Node):
    def __init__(self, units = 10, activation = tf.nn.relu):
        super().__init__(units, activation)

    def create_node(self) -> tf.keras.layers.Layer:
        return tf.keras.layers.Dense(units=self.units,
                                    activation=self.activation,
                                    # kernel_initializer=self.kernel_init,
                                    # bias_initializer=self.bias_init,
                                    dtype=self.dtype)

    def compile_model(self):
        learning_rate = 0.01
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

class DenseFlipout(Node):
    def __init__(self, units = 10, activation = tf.nn.relu):
        super().__init__(units, activation)
        self.dataset_size = 10
        dist = tfp.distributions
        self.kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(self.dataset_size, dtype=self.dtype))

    def create_node(self) -> tf.keras.layers.Layer: # tfp.layers.Layer???
        return tfp.layers.DenseFlipout(units=self.units,
                                    kernel_divergence_fn=self.kl_divergence_function,
                                    activation=self.activation,
                                    # kernel_initializer=self.kernel_init,
                                    # bias_initializer=self.bias_init,
                                    dtype=self.dtype)

class Blueprint():
    def __init__(self, genome, layers, input_shape, layerType) -> tf.keras.Model:
        self.model = None
        self.input_shape = input_shape
        self.dtype = tf.float32
        self.layerType = layerType
        self.layers = layers

        self.blueprint_graph = self.blueprint_convert(layers, genome)

        self.process_graph()

        learning_rate = 0.01
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    def blueprint_convert(self, layersNumber, x):
        blueprint_graph = []
        start = end = 0
        for i in range(layersNumber-1):
            start = end
            end = start+i+1
            blueprint_graph.append(x[start:end])

        return blueprint_graph

    def process_graph(self):
            input = tf.keras.Input(shape=self.input_shape, dtype=self.dtype)
            flatten = tf.keras.layers.Flatten()
            flat_input = flatten(input)

            layers = [self.layerType() for _ in range(len(self.blueprint_graph) + 1)]
            layerClass = self.layerType()   
            layer = layerClass.create_node()(flat_input)
            layers[0] = layer

            for idx, gene in enumerate(self.blueprint_graph, 1):
                if np.count_nonzero(gene) == 0:
                    layer = layers[0]
                else:
                    if np.count_nonzero(gene) > 1:
                        nonzero_genes = [layers[i] for i in np.nonzero(gene)[0]]
                        layer = tf.keras.layers.concatenate(nonzero_genes)
                    else:
                        layer = layers[np.nonzero(gene)[0][0]]
                layerClass = self.layerType()
                layer = layerClass.create_node()(layer)
                layers[idx] = layer

            # output_layers = [#{'class_name': 'Flatten', 'config': {}},
            #                 {'class_name': 'Dense', 'config': {'units': 10, 'activation': 'softmax'}}]
            # deserialized_output_layers = [tf.keras.layers.deserialize(layer_config) for layer_config in output_layers]
            
            # for out_layer in deserialized_output_layers:
            #     output = out_layer(layer)
            output = tf.keras.layers.Dense(units=10, activation='softmax')(layer)

            self.model = tf.keras.Model(input, output)

    def get_model(self) -> tf.keras.Model:
        return self.model
