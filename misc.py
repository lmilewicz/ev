import numpy as np
import tensorflow as tf

import node


def genome_convert(genome, layers_indexes):
    #   Converts genome into blueprint. E.g. 4 layers genome: [1, 0, 1, 1, 1, 1] -> [[1], [0, 1], [1, 1, 1]]
    return [genome[layers_indexes[i]:layers_indexes[i+1]] for i in range(len(layers_indexes)-1)]


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


def remove_disconnected_layers(X, config):
    _X = np.zeros(X.shape, dtype=np.int)

    for i in range(X.shape[0]):
        layers = np.zeros(config.n_layers*config.n_modules, dtype=np.int)

        for j in range(config.n_modules):
            genome_start = config.module_genome_len*j
            genome_end = config.module_genome_len*(j+1)
            
            genome_module = X[i, genome_start:genome_end]
            layers[genome_start] = 1

            genome_graph = genome_convert(genome_module, layers_indexes=config.layers_indexes)
            for idx, gene in enumerate(genome_graph, start=1):
                layer = 0
                gene_copy = gene.copy()
                if np.count_nonzero(gene) > 0:
                    for j in np.nonzero(gene)[0]:
                        if(layers[j] == 0): 
                            gene_copy[j] = 0
                        else:
                            layer = 1
                layers[idx] = layer

                index_1 = config.layers_indexes[idx-1] + genome_start
                index_2 = config.layers_indexes[idx] + genome_start
                _X[i, index_1:index_2] = gene_copy

    return _X
    

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


# def remove_disconnected_layers(X, config):
#     _X = np.zeros(X.shape, dtype=np.int)

#     for i in range(X.shape[0]):
#         layers = np.zeros(config.n_layers, dtype=np.int)
#         layers[0] = 1

#         genome_graph = genome_convert(X[i, :], layers_indexes=config.layers_indexes)
#         for idx, gene in enumerate(genome_graph, start=1):
#             layer = 0
#             gene_copy = gene.copy()
#             if np.count_nonzero(gene) > 0:
#                 for j in np.nonzero(gene)[0]:
#                     if(layers[j] == 0): 
#                         gene_copy[j] = 0
#                     else:
#                         layer = 1
#             layers[idx] = layer

#             _X[i, config.layers_indexes[idx-1]:config.layers_indexes[idx]] = gene_copy

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

