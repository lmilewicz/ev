# Plan:
# Create a class for layers;
# Create models based on x layers with BNN;
# Evaluate nn timing on MNIST and CIFAR -> TF vs PyTorch

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
    def __init__(self, blueprint_graph, input_shape, layerType):
        self.blueprint_graph = blueprint_graph
        self.model = None
        self.input_shape = input_shape
        self.dtype = tf.float32
        self.layerType = layerType

        self.process_graph()

        learning_rate = 0.01
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    def process_graph(self):
            input = tf.keras.Input(shape=self.input_shape, dtype=self.dtype)
            flatten = tf.keras.layers.Flatten()
            flat_input = flatten(input)

            layers = [self.layerType() for _ in range(len(self.blueprint_graph) + 1)]
            layerClass = self.layerType()   
            layer = layerClass.create_node()(flat_input)
            layers[0] = layer

            for idx, gene in enumerate(self.blueprint_graph, 1):
                if np.count_nonzero(gene) > 0:
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

n_epochs = 10
mnist_dataset = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = mnist_dataset
train_images, test_images = train_images / 255.0, test_images / 255.0


import time
start_time = time.time()

### Normal
blueprint_graph = [[1], [0, 1], [1, 1, 1]]
input_shape = (28, 28, 1)
layerType = DenseLayer
blueprint_object = Blueprint(blueprint_graph, input_shape, layerType)

model = blueprint_object.get_model()

model.fit(x=train_images,
            y=train_labels,
            epochs=n_epochs,
            use_multiprocessing=True,
            batch_size=64)

print('Execution Time: %s' % (time.time()-start_time))

model.summary()

### TFP
# model_tfp = get_model((28, 28, 1), dtype=dtype, layerType=DenseFlipout)

# model_tfp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
# model_tfp.fit(x=train_images,
#             y=train_labels,
#             epochs=10,
#             use_multiprocessing=True,
#             batch_size=64)

# model_tfp.summary()