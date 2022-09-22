import numpy as np
import tensorflow as tf

import node


def module_convert(genome, layers_indexes):
    #   Converts module into blueprint. E.g. 4 layers module: [1, 0, 1, 1, 1, 1] -> [[1], [0, 1], [1, 1, 1]]
    return [genome[layers_indexes[i]:layers_indexes[i+1]] for i in range(len(layers_indexes)-1)]


def genome_convert(genome, module_layers_indexes):
    #   Converts genome into blueprint
    genome_converted = []
    module_start = 0
    for layers_indexes in module_layers_indexes:
        genome_converted.append(module_convert(genome[module_start:module_start+layers_indexes[-1]].tolist(), layers_indexes))
        module_start = module_start + layers_indexes[-1]
    genome_converted[-1].append([genome[-1]])

    return genome_converted


def get_params_dict(config, layer_type):
	params_dict = {'activation':	config.activation,
					'dtype':		tf.float32,
					'prob_layer':	False}
	if layer_type == node.DenseLayer:
		params_dict['units'] = config.units
	elif layer_type == node.Convolution2D:
		params_dict['kernel_size'] = config.kernel_size
	else:
		raise ValueError('In get_params_dict: layer_type with wrong type: '+str(type(layer_type)))
	return params_dict


class RemoveDisconnectedLayers():
    def __init__(self, X, config) -> None:
        self._X = np.zeros(X.shape, dtype=np.int)
        self.X = X
        self.config = config

        self.main()


    def main(self):
        for i in range(self.X.shape[0]):
            self.process_module(i,\
                start_idx=0,\
                n_modules=self.config.max_n_conv_modules,\
                n_layers=self.config.max_n_conv_layers,\
                module_genome_len=self.config.conv_module_genome_len,\
                layers_indexes=self.config.conv_layers_indexes)

            self.process_module(i,\
                start_idx=self.config.max_n_conv_modules*self.config.conv_module_genome_len,\
                n_modules=self.config.max_n_ann_modules,\
                n_layers=self.config.max_n_ann_layers,\
                module_genome_len=self.config.ann_module_genome_len,\
                layers_indexes=self.config.ann_layers_indexes)


    def process_module(self, i, start_idx, n_modules, n_layers, module_genome_len, layers_indexes):
        for j in range(n_modules):
            activated_layers_array = np.zeros(n_layers)
            activated_layers_array[0] = 1

            genome_start = module_genome_len*j + start_idx
            genome_end = module_genome_len*(j+1) + start_idx
                
            genome_module = self.X[i, genome_start:genome_end]
            genome_graph = module_convert(genome_module, layers_indexes)
            for idx, gene in enumerate(genome_graph, start=1):
                gene_copy = gene.copy()
                gene_copy.resize(n_layers)

                # print(idx, "xd3")
                # print(gene_copy)
                # print(activated_layers_array)

                gene_copy = np.multiply(gene_copy, activated_layers_array)
                # print(gene_copy)


                if sum(gene_copy) > 0:
                    activated_layers_array[idx] = 1  
                
                index_1 = layers_indexes[idx-1] + genome_start
                index_2 = layers_indexes[idx] + genome_start

                gene_copy2 = gene_copy[0:len(gene)].astype(np.int)
                self._X[i, index_1:index_2] = gene_copy2


    def return_new_X(self):
        return self._X



def get_flops(model):
    ''' Returns GFLOPS value needed to process neural network model
    ISSUE: It creates WARNINGS with tensorflow
    '''
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = tf.python.framework.convert_to_constants.convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'

        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops* 1e-9


def get_params_number(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams

    return totalParams* 1e-9


def get_graph(module):
    graph = {}
    graph[1] = []

    module_len = len(module)
    for i in range(module_len):
        if i < module_len-1 and len(module[i]) >= len(module[i+1]):
            module_len = module_len - 1 
            break 

    for i in range(module_len):
        graph[i + 2] = [j + 1 for j in range(len(module[i])) if module[i][j] == 1]
    
    graph[module_len+2] = [out for (out, input) in graph.items() if any(i==1 for i in input)]

    return graph


# def old_remove_disconnected_layers(X, config):
def get_best_genome(algorithm, config):
    pop_obj = algorithm.pop.get('F')
    X = algorithm.pop.get('X')
    best_index = np.argmin(pop_obj[:, 0])
    best_genome = genome_convert(X[best_index, :], [config.conv_layers_indexes, config.ann_layers_indexes])
    return best_genome

# def old_remove_disconnected_layers(X, config):
#     _X = np.zeros(X.shape, dtype=np.int)

#     for i in range(X.shape[0]):
#         layers = np.zeros(config.n_layers*config.max_n_modules, dtype=np.int)

#         for j in range(config.max_n_modules):
#             genome_start = config.module_genome_len*j
#             genome_end = config.module_genome_len*(j+1)
            
#             genome_module = X[i, genome_start:genome_end]
#             layers[genome_start] = 1

#             genome_graph = module_convert(genome_module, layers_indexes=config.layers_indexes)
#             for idx, gene in enumerate(genome_graph, start=1):
#                 layer = 0
#                 gene_copy = gene.copy()
#                 if np.count_nonzero(gene) > 0:
#                     for j in np.nonzero(gene)[0]:
#                         if(layers[j] == 0): 
#                             gene_copy[j] = 0
#                         else:
#                             layer = 1
#                 layers[idx] = layer

#                 index_1 = config.layers_indexes[idx-1] + genome_start
#                 index_2 = config.layers_indexes[idx] + genome_start
#                 _X[i, index_1:index_2] = gene_copy

#     return _X


# def remove_disconnected_layers(X, config):
#     _X = np.zeros(X.shape, dtype=np.int)

#     for i in range(X.shape[0]):
#         layers = np.zeros(config.n_layers*config.max_n_modules, dtype=np.int)

#         for j in range(config.max_n_modules):
#             genome_start = config.module_genome_len*j
#             genome_end = config.module_genome_len*(j+1)
            
#             genome_module = X[i, genome_start:genome_end]
#             layers[genome_start] = 1

#             genome_graph = module_convert(genome_module, layers_indexes=config.layers_indexes)
#             for idx, gene in enumerate(genome_graph, start=1):
#                 layer = 0
#                 gene_copy = gene.copy()
#                 if np.count_nonzero(gene) > 0:
#                     for j in np.nonzero(gene)[0]:
#                         if(layers[j] == 0): 
#                             gene_copy[j] = 0
#                         else:
#                             layer = 1
#                 layers[idx] = layer

#                 index_1 = config.layers_indexes[idx-1] + genome_start
#                 index_2 = config.layers_indexes[idx] + genome_start
#                 _X[i, index_1:index_2] = gene_copy

#     return _X



# def get_flops(model):
#     ''' Returns GFLOPS value needed to process neural network model
#     '''
#     batch_size = 1
#     inputs = [tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs]
#     real_model = tf.function(model).get_concrete_function(inputs)
#     frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

#     run_meta = tf.compat.v1.RunMetadata()
#     opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#     opts['output'] = 'none'

#     flops = tf.compat.v1.profiler.profile(
#         graph=frozen_func.graph, run_meta=run_meta, cmd='scope', options=opts
#     )
#     return float(flops.total_float_ops) * 1e-9

