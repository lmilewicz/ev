import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation

from misc import remove_disconnected_layers, NewRemoveDisconnectedLayers



class SamplingAll(Sampling):
    def __init__(self) -> None:
        super().__init__()
        
    def _do(self, problem, n_samples, **kwargs):
        if problem.config.load_genomes != None:
            _X = np.array(problem.config.load_genomes.copy())
        else:
            _X = np.random.random((n_samples, problem.n_var))
            _X = (_X > 0.5).astype(np.int)

        # return remove_disconnected_layers(val, problem.config)
        return NewRemoveDisconnectedLayers(_X, problem.config).return_new_X()

class SamplingFromSmall(Sampling):
    def __init__(self) -> None:
        super().__init__()
        
    def _do(self, problem, n_samples, **kwargs):
        if problem.config.load_genomes != None:
            _X = np.array(problem.config.load_genomes.copy())
        else:
            _X = np.zeros((n_samples, problem.n_var))
            _X[:,0] = np.random.random(n_samples)
            _X = (_X > 0.5).astype(np.int)

        # return remove_disconnected_layers(val, problem.config)
        return NewRemoveDisconnectedLayers(_X, problem.config).return_new_X()


class MutationAll(Mutation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        if problem.config.enable_xgboost:
            _X = xgboost_mutation(X, self.prob)
        else:
            _X = no_xgboost_mutation(X, self.prob)

        # return remove_disconnected_layers(_X, problem.config)
        return NewRemoveDisconnectedLayers(_X, problem.config).return_new_X()


class MutationFromSmall(Mutation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var
        
        _X = np.full(X.shape, 0)

        for j in range(X.shape[0]):
            break_loop = 0
            for i in range(len(problem.config.layers_indexes)-1):
                start = problem.config.layers_indexes[i]
                end = problem.config.layers_indexes[i+1]
                X_layer = X[j, start:end].copy()
                _X_layer = X[j, start:end].copy()
                if(np.sum(X_layer))==0:
                    _X_layer[end-start-1] = (1 if np.random.random() > self.prob else 0)
                    break_loop = 1
                else:
                    M = np.random.random(X_layer.shape)
                    flip, no_flip = M < self.prob, M >= self.prob
                    _X_layer[flip] = np.logical_not(X_layer[flip]).astype(np.int)
                    _X_layer[no_flip] = X_layer[no_flip].astype(np.int)
                    if(np.sum(X_layer)==0):
                        break_loop=1
                        _X[j, end:X.shape[1]] = np.zeros(X.shape[1]-end)
                _X[j, start:end] = _X_layer

                if(break_loop): break

        # return remove_disconnected_layers(_X, problem.config)
        return NewRemoveDisconnectedLayers(_X, problem.config).return_new_X()



def xgboost_mutation(X, prob):
    X_without_output = X[:,:-1].copy().astype(np.bool)
    X_output = X[:,-1].copy()

    _X = np.full(X_without_output.shape, np.inf)

    M = np.random.random(X_without_output.shape)
    flip, no_flip = M < prob, M >= prob

    _X[flip] = np.logical_not(X_without_output[flip])
    _X[no_flip] = X_without_output[no_flip]

    _X = _X.astype(np.int)

    ### Output layer mutation::: ###
    X_output = X_output.reshape(len(X_output),1)
    M = np.random.random(X_output.shape)
    flip, no_flip = M < prob/2, M >= prob/2

    X_output[flip] = np.mod(X_output[flip]+1, 3)
    X_output[no_flip] = X_output[no_flip]
    #################################

    _X = np.append(_X, X_output, axis=1)


def no_xgboost_mutation(X, prob):
    X = X.astype(np.bool)
    _X = np.full(X.shape, np.inf)

    M = np.random.random(X.shape)
    flip, no_flip = M < prob, M >= prob

    _X[flip] = np.logical_not(X[flip])
    _X[no_flip] = X[no_flip]
    _X = _X.astype(np.int)

    return _X