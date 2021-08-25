import tensorflow as tf
import numpy as np

class Blueprint():
    def __init__(self, genome, config, dataset_size) -> tf.keras.Model:
        self.config = config
        self.dataset_size = dataset_size
        self.model = None
        self.dtype = tf.float32

        self.blueprint_graph = self.blueprint_convert(layersNumber=config.max_layers, x=genome)
        self.process_graph()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    def blueprint_convert(self, layersNumber, x):
        blueprint_graph = []
        start = end = 0
        for i in range(layersNumber-1):
            start = end
            end = start+i+1
            blueprint_graph.append(x[start:end])

        return blueprint_graph

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
                if np.count_nonzero(gene) == 0:
                    layer = None
                else:
                    if np.count_nonzero(gene) > 1:
                        nonzero_genes = [layers[i] for i in np.nonzero(gene)[0]]
                        layer = tf.keras.layers.concatenate(nonzero_genes)
                    else:
                        layer = layers[np.nonzero(gene)[0][0]]
                    if(layer == None):
                        print("AAAAAAAAAAAAAAAAAAAAAA")
                    else:
                        layerClass = config.layerType(units = config.units, activation = config.activation)
                        layer = layerClass.create_node(self.dataset_size)(layer)
                layers[idx] = layer
            
            ### OLD version:::
            # for idx, gene in enumerate(self.blueprint_graph, start=1):
            #     if np.count_nonzero(gene) == 0:
            #         layer = layer[idx-1]
            #     else:
            #         if np.count_nonzero(gene) > 1:
            #             nonzero_genes = [layers[i] for i in np.nonzero(gene)[0]]
            #             layer = tf.keras.layers.concatenate(nonzero_genes)
            #         else:
            #             layer = layers[np.nonzero(gene)[0][0]]
            #     layerClass = config.layerType(units = config.units, activation = config.activation)
            #     layer = layerClass.create_node(self.dataset_size)(layer)
            #     layers[idx] = layer

            output = tf.keras.layers.Dense(units=config.out_units, activation=config.out_activation)(layer)

            self.model = tf.keras.Model(input, output)

    def get_model(self) -> tf.keras.Model:
        return self.model
