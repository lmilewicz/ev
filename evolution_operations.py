import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation

from misc import remove_disconnected_layers


class SamplingAll(Sampling):
    def __init__(self) -> None:
        super().__init__()
        
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        val = (val > 0.5).astype(np.int)
        return remove_disconnected_layers(val, problem.config)


class SamplingFromSmall(Sampling):
    def __init__(self) -> None:
        super().__init__()
        
    def _do(self, problem, n_samples, **kwargs):
        val = np.zeros((n_samples, problem.n_var))
        val[:,0] = np.random.random(n_samples)
        val = (val > 0.5).astype(np.int)
        return remove_disconnected_layers(val, problem.config)


class MutationAll(Mutation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        X_without_output = X[:,:-1].copy().astype(np.bool)
        X_output = X[:,-1].copy()

        _X = np.full(X_without_output.shape, np.inf)

        M = np.random.random(X_without_output.shape)
        flip, no_flip = M < self.prob, M >= self.prob

        _X[flip] = np.logical_not(X_without_output[flip])
        _X[no_flip] = X_without_output[no_flip]

        _X = _X.astype(np.int)

        ### Output layer mutation::: ###
        X_output = X_output.reshape(len(X_output),1)
        M = np.random.random(X_output.shape)
        flip, no_flip = M < self.prob/2, M >= self.prob/2

        X_output[flip] = np.mod(X_output[flip]+1, 3)
        X_output[no_flip] = X_output[no_flip]
        #################################

        _X = np.append(_X, X_output, axis=1)
        return remove_disconnected_layers(_X, problem.config)

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

        return remove_disconnected_layers(_X, problem.config)
