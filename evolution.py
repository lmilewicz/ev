import numpy as np
import tensorflow as tf

from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.model.mutation import Mutation

from blueprint import Blueprint
from data_loader import DataLoader

def do_every_generations(algorithm):
    gen = algorithm.n_gen
    pop_obj = algorithm.pop.get("F")

    print("generation = {}".format(gen))
    print("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    
class EVProblem(Problem):
    def __init__(self, config):
        max_connections = int((config.max_layers-1)*config.max_layers/2) 
        # E.g. genome for 4 layers  - all connected: [1], [1, 1], [1, 1, 1] -> 6

        super().__init__(n_var=max_connections, n_obj=config.n_obj, n_constr=config.n_constr, type_var=np.int)

        self.config = config
        self.max_layers = config.max_layers
        self.xl = np.zeros(max_connections)
        self.xu = np.ones(max_connections)

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = DataLoader(config.dataset)

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            blueprint_object = Blueprint(genome=x, config=self.config, dataset_size = len(self.train_images))
            model = blueprint_object.get_model()
            
            model.fit(x=self.train_images,
                        y=self.train_labels,
                        epochs=self.config.n_epochs,
                        use_multiprocessing=True,
                        batch_size=self.config.batch_size,
                        verbose=0)

            performance = model.evaluate(self.test_images, self.test_labels, verbose=0)

            objs[i, 0] = 1 - performance

        out["F"] = objs


class SamplingAll(Sampling):
    def __init__(self) -> None:
        super().__init__()
        
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val > 0.5).astype(np.int)

class SamplingFromSmall(Sampling):
    def __init__(self) -> None:
        super().__init__()
        
    def _do(self, problem, n_samples, **kwargs):
        val = np.zeros((n_samples, problem.n_var))
        val[:,0] = np.random.random(n_samples)
        return (val > 0.5).astype(np.int)



class MutationAll(Mutation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        X = X.astype(np.bool)
        _X = np.full(X.shape, np.inf)

        M = np.random.random(X.shape)
        flip, no_flip = M < self.prob, M >= self.prob

        _X[flip] = np.logical_not(X[flip])
        _X[no_flip] = X[no_flip]

        return _X

class MutationFromSmall(Mutation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var
        
        _X = np.full(X.shape, 0)

        for j in range(X.shape[0]):
            break_loop = start = end = 0
            for i in range(problem.max_layers-1):
                start = end
                end = start+i+1
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

        return _X

