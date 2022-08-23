import tensorflow as tf
import numpy as np
import misc


class Module():
	def __init__(self, genome, config, layer_type, layers_indexes, input_layer):
		self.layer_type = layer_type
		self.module_graph = misc.module_convert(genome, layers_indexes)
		self.process_graph(input_layer, config)

	def process_graph(self, input_layer, config):
		layers = [None] * (len(self.module_graph) + 1)
		layer = input_layer
		layers[0] = layer

		params_dict = misc.get_params_dict(config, self.layer_type)
		layer_object = self.layer_type(params_dict)   

		for idx, gene in enumerate(self.module_graph, start=1):
			layer = None
			if np.count_nonzero(gene) > 0:
				if np.count_nonzero(gene) > 1:
					nonzero_genes = [layers[i] for i in np.nonzero(gene)[0]]
					layer = tf.keras.layers.concatenate(nonzero_genes)
				else:
					layer = layers[np.nonzero(gene)[0][0]]
				if(layer is None):
					raise ValueError('In Module: layer should have never been None if we use remove_disconected_layers!!!')
				else:
					layer = layer_object.create_node()(layer)
				layers[idx] = layer

		last_active_layer = None
		for layer in layers:
			if(layer is not None): last_active_layer = layer

		if(last_active_layer is None):
				raise ValueError('In Module: last_active_layer should have never been None!!!')
		else:
			self.module = last_active_layer

		
	def get_module(self):
		return self.module
