import tensorflow as tf
import numpy as np
import misc


class Module():
	def __init__(self, genome, config, layer_type, layers_indexes, input_layer):
		self.layer_type = layer_type
		self.old_module_graph = misc.module_convert(genome, layers_indexes)
		module_genome = misc.module_convert(genome, layers_indexes)
		self.module_graph = misc.get_graph(module_genome)

		self.process_graph(input_layer, config)

	def process_graph(self, input_layer, config):
		layers = [None] * (len(self.module_graph) + 1)		
		old_layers = [None] * (len(self.module_graph) + 1)

		layer = input_layer		
		old_layer = input_layer

		layers[0] = layer
		old_layers[0] = layer

		params_dict = misc.get_params_dict(config, self.layer_type)
		layer_object = self.layer_type(params_dict)   
		old_layer_object = self.layer_type(params_dict)   

		print(self.old_module_graph)
		print(self.module_graph)

		for idx, gene in enumerate(self.old_module_graph, start=1):
			old_layer = None
			if np.count_nonzero(gene) > 0:
				if np.count_nonzero(gene) > 1:
					nonzero_genes = [old_layers[i] for i in np.nonzero(gene)[0]]
					old_layer = tf.keras.layers.concatenate(nonzero_genes)
				else:
					old_layer = old_layers[np.nonzero(gene)[0][0]]
				if(old_layer is None):
					raise ValueError('In Old Module: layer should have never been None if we use remove_disconected_layers!!!')
				else:
					old_layer = old_layer_object.create_node()(old_layer)
				old_layers[idx] = old_layer

		layers[1] = layer_object.create_node()(layer)
		for output, inputs in self.module_graph.items():
			layer = None
			if len(inputs) == 0: continue
			if len(inputs) > 1:
				layer = tf.keras.layers.concatenate(inputs)
			else:
				layer = layers[inputs[0]]
			if(layer is None):
				raise ValueError('In Module: layer should have never been None if we use remove_disconected_layers!!!')
			else:
				layer = layer_object.create_node()(layer)

			layers[idx] = layer

		print(layers)
		print(old_layers)



		last_active_layer = None
		for layer in layers:
			if(layer is not None): last_active_layer = layer

		if(last_active_layer is None):
				raise ValueError('In Module: last_active_layer should have never been None!!!')
		else:
			self.module = last_active_layer

		
	def get_module(self):
		return self.module
