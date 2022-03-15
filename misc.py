import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def blueprint_convert(genome, layers_indexes):
    #   Converts genome into blueprint. E.g. 4 layers genome: [1, 0, 1, 1, 1, 1] -> [[1], [0, 1], [1, 1, 1]]
    return [genome[layers_indexes[i]:layers_indexes[i+1]] for i in range(len(layers_indexes)-1)]

def remove_disconnected_layers(X, config):
    _X = np.zeros(X.shape, dtype=np.int)

    for i in range(X.shape[0]):
        layers = np.zeros(config.max_layers, dtype=np.int)
        layers[0] = 1

        blueprint_graph = blueprint_convert(X[i, :], layers_indexes=config.layers_indexes)
        for idx, gene in enumerate(blueprint_graph, start=1):
            layer = 0
            gene_copy = gene.copy()
            if np.count_nonzero(gene) > 0:
                for j in np.nonzero(gene)[0]:
                    if(layers[j] == 0): 
                        gene_copy[j] = 0
                    else:
                        layer = 1
            layers[idx] = layer

            _X[i, config.layers_indexes[idx-1]:config.layers_indexes[idx]] = gene_copy

    return _X


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
#         graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
#     )
#     return float(flops.total_float_ops) * 1e-9


def get_flops(model):
    ''' Returns GFLOPS value needed to process neural network model
    ISSUE: It creates WARNINGS with tensorflow
    '''
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'

        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops* 1e-9

def get_params_number(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams

    return totalParams* 1e-9