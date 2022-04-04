import tensorflow as tf
import tensorflow_probability as tfp

from module import Module
import node

class Blueprint():
    def __init__(self, genome, config) -> tf.keras.Model:
        self.model = None
        self.modules_list = []
        if config.debug: print('Blueprint:', genome)
        
        tf.keras.backend.clear_session()
        
        input = tf.keras.Input(shape=config.input_shape, dtype=config.dtype)
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

        if genome[-1] == 0:
            output = tf.keras.layers.Dense(
                        units=config.out_units, 
                        name='out_dense_layer',
                        activation=config.out_activation,
                        dtype=config.dtype)(last_layer)
        elif genome[-1] == 1:
            output = tfp.layers.DenseFlipout(units=config.out_units,
                                    name='out_prob_layer',
                                    kernel_divergence_fn=config.kl_divergence_function,
                                    activation=config.out_activation,
                                    dtype=config.dtype)(last_layer)
        # elif genome[-1] == 2:
        #     TO DO --> XGboost
        else:
            raise ValueError('In Blueprint: undefined output: '+str(genome[-1]))


        self.model = tf.keras.Model(input, output)

        if config.debug: print(self.model.summary())

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


    def get_model(self) -> tf.keras.Model:
        return self.model
