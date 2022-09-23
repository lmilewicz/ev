import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation

from misc import RemoveDisconnectedLayers


class SamplingFromSmall(Sampling):
    def __init__(self) -> None:
        super().__init__()
        
    def _do(self, problem, n_samples, **kwargs):
        if problem.config.load_genomes != None:
            _X = np.array(problem.config.load_genomes.copy())
        else:
            _X = np.zeros((n_samples, problem.n_var))
            R = calculate_range(_X.shape, problem.config)
            _X = np.random.random((n_samples, problem.n_var))
            _X[~R] = 0
            _X = (_X > 0.5).astype(np.int)

        return RemoveDisconnectedLayers(_X, problem.config).return_new_X()


class MutationFromSmall(Mutation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):       
        _X = perform_mutations(X, problem.config)

        return RemoveDisconnectedLayers(_X, problem.config).return_new_X()


def perform_mutations(X, config):
    _X = X.copy()

    topology_range_mutation(config)
    R = calculate_range(X.shape, config)

    dice_array = np.random.random(X.shape[0])

    # Topology
    topology_mask = np.full(X.shape, False)
    topology_mask[dice_array < 0.5, :] = True
    topology_mask[~R] = False
    _X = topology_mutation(X, prob=1.0/config.topology_len, R=topology_mask)

    # Neurons number & Activation function
    module_index = config.topology_len\
        + np.random.randint(config.max_n_modules, size=X.shape[0])*2

    neurons_mask = np.full(X.shape, False)
    activation_mask = np.full(X.shape, False)

    search = np.where(np.logical_and(dice_array>=0.5, dice_array<0.7))[0]
    if search.size: neurons_mask[search, module_index] = True
    search = np.where(np.logical_and(dice_array>=0.7, dice_array<0.9))[0]
    if search.size: activation_mask[search, module_index+1] = True

    _X[neurons_mask] = np.random.randint(7)
    _X[activation_mask] = np.random.randint(3)

    # Output module
    output_mask = np.full(X.shape, False)
    output_mask[dice_array >= 0.9, -1] = True

    _X[output_mask] = np.random.randint(3)

    return _X


def topology_mutation(X, prob, R):
    _X = np.full(X.shape, np.inf)

    M = np.random.random(X.shape)
    flip, no_flip = M < prob, M >= prob

    _X[flip] = np.logical_not(X[flip])
    _X[no_flip] = X[no_flip]
    _X[~R] = False

    _X = _X.astype(np.int)

    return _X

def topology_range_mutation(config):
    module_update_prob = 0.1
    layer_update_prob = 0.25

    dice = np.random.random()
    if dice < module_update_prob:
        if(dice < module_update_prob/4): 
            if(config.n_conv_modules < config.max_n_conv_modules): config.n_conv_modules = config.n_conv_modules + 1
        elif(dice < module_update_prob/2): 
            if(config.n_conv_modules> 0): config.n_conv_modules = config.n_conv_modules - 1
        elif(dice < module_update_prob*3/4): 
            if(config.n_ann_modules < config.max_n_ann_modules): config.n_ann_modules = config.n_ann_modules + 1
        else: 
            if(config.n_ann_modules): config.n_ann_modules = config.n_ann_modules - 1

    dice = np.random.random()
    if dice < layer_update_prob:
        if(dice < layer_update_prob/4): 
            if(config.n_conv_layers < config.max_n_conv_layers): config.n_conv_layers = config.n_conv_layers + 1
        elif(dice < layer_update_prob/2): 
            if(config.n_conv_layers > 0): config.n_conv_layers = config.n_conv_layers - 1
        elif(dice < layer_update_prob*3/4): 
            if(config.n_ann_layers < config.max_n_ann_layers): config.n_ann_layers = config.n_ann_layers + 1
        else: 
            if(config.n_ann_layers > 0): config.n_ann_layers = config.n_ann_layers - 1



def calculate_range(shape, config):
    A = np.full(shape, False)
    conv_module_len = int(config.n_conv_layers*(config.n_conv_layers-1)*0.5)
    max_conv_module_len = int(config.max_n_conv_layers*(config.max_n_conv_layers-1)*0.5)
    ann_module_len = int(config.n_ann_layers*(config.n_ann_layers-1)*0.5)
    max_ann_module_len = int(config.max_n_ann_layers*(config.max_n_ann_layers-1)*0.5)

    start_array = 0
    for i in range(config.max_n_conv_modules):
        if i < config.n_conv_modules: A[:, start_array:conv_module_len+start_array] = True
        start_array = start_array + max_conv_module_len

    for i in range(config.max_n_ann_modules):
        if i < config.n_ann_layers: A[:, start_array:ann_module_len+start_array] = True
        start_array = start_array + max_ann_module_len

    return A



import evolution
from config import Config

import sys
np.set_printoptions(threshold=sys.maxsize)


if __name__ == '__main__':

    mutation = MutationFromSmall()
    config = Config()
    problem = evolution.EVProblem(config)
    X = np.full((3, config.genome_len), 1)

    print(X)
    print(mutation._do(problem, X))
