import tensorflow as tf
import numpy as np

from module import Module
import node

class Blueprint():
    def __init__(self, genome, config) -> tf.keras.Model:
        self.model = None
        self.modules_list = []
        if config.debug: print('Blueprint:', genome)
        
        tf.keras.backend.clear_session()

        input = tf.keras.Input(shape=config.input_shape, dtype=tf.float32)
        layer_type = node.Convolution2D

        last_layer = input
        
        for i in range(config.n_modules):

            if i == config.n_conv_modules:
                last_layer = node.MaxPool2D()(last_layer)
                last_layer = tf.keras.layers.Flatten()(last_layer)
                layer_type = node.DenseLayer

            genome_module = genome[config.module_genome_len*i:config.module_genome_len*(i+1)]
            module = Module(genome_module, config, layer_type, input_layer = last_layer)

            last_layer = module.get_module()

        output = tf.keras.layers.Dense(
                units=config.out_units, 
                name='out_layer',
                activation=config.out_activation)(last_layer)


        self.model = tf.keras.Model(input, output)

        if config.debug: print(self.model.summary())

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


    def get_model(self) -> tf.keras.Model:
        return self.model
