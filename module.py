import tensorflow as tf
import numpy as np
import misc


class Module():
	def __init__(self, genome, config, layer_type, layers_indexes, input_layer):
		self.layer_type = layer_type
		module_genome = misc.module_convert(genome, layers_indexes)
		self.module_graph = misc.get_graph(module_genome)

		self.process_graph(input_layer, config)

	def process_graph(self, input_layer, config):
		layers = [None] * (len(self.module_graph) + 1)		
		layer = input_layer		
		layers[0] = layer

		params_dict = misc.get_params_dict(config, self.layer_type)
		layer_object = self.layer_type(params_dict)   

		layers[1] = layer_object.create_node()(layer)
		for output, inputs in self.module_graph.items():
			layer = None
			if len(inputs) == 0: continue
			if len(inputs) > 1:
				nonzero_genes = [layers[i] for i in inputs]
				layer = tf.keras.layers.concatenate(nonzero_genes)
			else:
				layer = layers[inputs[0]]
			if(layer is None):
				raise ValueError('In Module: layer should have never been None if we use remove_disconected_layers!!!')
			else:
				if output != list(self.module_graph)[-1]:
					layer = layer_object.create_node()(layer)

			layers[output] = layer

		self.module = layers[-1]

		
	def get_module(self):
		return self.module
