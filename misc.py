import numpy as np

def blueprint_convert(genome, layers_indexes):
    #   Converts genome into blueprint. E.g. 4 layers genome: [1, 0, 1, 1, 1, 1] -> [[1], [0, 1], [1, 1, 1]]
    return [genome[layers_indexes[i]:layers_indexes[i+1]] for i in range(len(layers_indexes)-1)]

def remove_disconnected_layers(X, config):
    _X = np.zeros(X.shape, dtype=np.int)

    for i in range(X.shape[0]):
        layers = np.zeros(config.max_layers, dtype=np.int)
        layers[0] = 1

        # print(X[i, :])
        # print(_X[i, :], 'yyy')

        blueprint_graph = blueprint_convert(X[i, :], layers_indexes=config.layers_indexes)
        for idx, gene in enumerate(blueprint_graph, start=1):
            layer = 0
            gene_copy = gene.copy()
            if np.count_nonzero(gene) > 0:
                # print(np.nonzero(gene))
                for j in np.nonzero(gene)[0]:
                    if(layers[j] == 0): 
                        gene_copy[j] = 0
                    else:
                        layer = 1
            layers[idx] = layer
            # print(_X[i, config.layers_indexes[idx-1]:config.layers_indexes[idx]])
            # print(gene_copy)
            # print(idx, layer, gene_copy, _X[i, config.layers_indexes[idx-1]:config.layers_indexes[idx]])

            _X[i, config.layers_indexes[idx-1]:config.layers_indexes[idx]] = gene_copy

        # print(_X[i, :], 'xxx')

    return _X