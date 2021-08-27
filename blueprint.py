import tensorflow as tf
import numpy as np

from misc import blueprint_convert

class Blueprint():
    def __init__(self, genome, config, dataset_size) -> tf.keras.Model:
        self.config = config
        self.dataset_size = dataset_size
        self.model = None
        self.dtype = tf.float32

        self.blueprint_graph = blueprint_convert(genome, layers_indexes=config.layers_indexes)
        self.process_graph()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        
    def process_graph(self):
            tf.keras.backend.clear_session()
            config = self.config
            
            input = tf.keras.Input(shape=config.input_shape, dtype=self.dtype)
            flatten = tf.keras.layers.Flatten()
            flat_input = flatten(input)

            layers = [config.layerType(units=config.units, activation=config.activation) for _ in range(len(self.blueprint_graph) + 1)]
            layerClass = config.layerType(units=config.units, activation=config.activation)   
            layer = layerClass.create_node(self.dataset_size)(flat_input)
            layers[0] = layer

            for idx, gene in enumerate(self.blueprint_graph, start=1):
                layer = None
                if np.count_nonzero(gene) > 0:
                    if np.count_nonzero(gene) > 1:
                        nonzero_genes = [layers[i] for i in np.nonzero(gene)[0]]
                        layer = tf.keras.layers.concatenate(nonzero_genes)
                    else:
                        layer = layers[np.nonzero(gene)[0][0]]
                    if(layer is None):
                        raise ValueError("In Blueprint: layer should have never been None if we use remove_disconected_layers!!!")
                    else:
                        layerClass = config.layerType(units = config.units, activation = config.activation)
                        layer = layerClass.create_node(self.dataset_size)(layer)
                layers[idx] = layer

            last_active_layer = None
            for layer in layers:
                if(layer == None): break
                else: last_active_layer = layer

            if(last_active_layer==None):
                raise ValueError("In Blueprint: last_active_layer should have never been None!!!")
            else:
                output = tf.keras.layers.Dense(units=config.out_units, activation=config.out_activation)(last_active_layer)
                self.model = tf.keras.Model(input, output)

    def get_model(self) -> tf.keras.Model:
        return self.model
