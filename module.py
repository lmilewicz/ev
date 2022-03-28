import tensorflow as tf
import numpy as np
from misc import genome_convert

class Module():
	def __init__(self, genome, config, layer_type, input_layer):
		self.layer_type = layer_type
		self.module_graph = genome_convert(genome, layers_indexes=config.layers_indexes)
		
		self.process_graph(input_layer, config)

	def process_graph(self, input_layer, config):
			layers = [None] * (len(self.module_graph) + 1)
			layer_type = self.layer_type(units=config.units, activation=config.activation)   
			layer = input_layer
			layers[0] = layer

			for idx, gene in enumerate(self.module_graph, start=1):
					layer = None
					if np.count_nonzero(gene) > 0:
						if np.count_nonzero(gene) > 1:
								nonzero_genes = [layers[i] for i in np.nonzero(gene)[0]]
								layer = tf.keras.layers.concatenate(nonzero_genes)
						else:
								layer = layers[np.nonzero(gene)[0][0]]
						if(layer is None):
								raise ValueError('In Blueprint: layer should have never been None if we use remove_disconected_layers!!!')
						else:
								layer = layer_type.create_node(config.dataset_size)(layer)
						layers[idx] = layer

			last_active_layer = None
			for layer in layers:
				if(layer is not None): last_active_layer = layer

			if(last_active_layer is None):
					raise ValueError('In Blueprint: last_active_layer should have never been None!!!')
			else:
				self.module = last_active_layer

	def get_module(self):
		return self.module