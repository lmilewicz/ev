import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation

from blueprint import Blueprint
from misc import get_params_number, genome_convert, remove_disconnected_layers

from config import Config

import time
from statistics import mean


class EVProblem(Problem):
    def __init__(self, config):
        # E.g. genome for 4 layers  - all connected: [1], [1, 1], [1, 1, 1] -> 6
        super().__init__(n_var=config.genome_len, n_obj=config.pop_size, 
            n_constr=config.n_constr, type_var=np.int)

        self.config = config
        self.xl = np.zeros(config.genome_len)
        self.xu = np.ones(config.genome_len)

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        best_perf = 0

        for i in range(x.shape[0]):
            time1 = time.time()

            blueprint_object = Blueprint(genome=x[i, :], config=self.config)
            model = blueprint_object.get_model()

            time2 = time.time()
            self.config.blueprint_time.append(time2-time1)
            history = model.fit(self.config.ds_train,
                        epochs=self.config.n_epochs,
                        use_multiprocessing=True,
                        batch_size=self.config.batch_size,
                        validation_data=self.config.ds_test,
                        verbose=0)

            self.config.fit_time.append(time.time()-time2)

            performance = history.history['val_sparse_categorical_accuracy'][-1]
            objs[i, 0] = 1 - performance
            objs[i, 1] = get_params_number(model)

            if self.config.debug and performance > best_perf:
                best_perf = performance
                best_model = model

        print('Timestats: blueprint_time: '
                +str(round(mean(self.config.blueprint_time), 2))
                +'. fit_time: '
                +str(round(mean(self.config.fit_time), 2)))
        
        if self.config.debug:
            print('Best perf: '+str(round(best_perf, 4)))
            best_model.summary()

        out['F'] = objs


def do_every_generations(algorithm):
    gen = algorithm.n_gen
    pop_obj = algorithm.pop.get('F')
    X = algorithm.pop.get('X')
    print('Generation = {}'.format(gen))
    print('population error: best = {:.3f}, mean = {:.3f}, median = {:.3f}, worst = {:.3f}'.format(
        np.min(pop_obj[:, 0]), 
        np.mean(pop_obj[:, 0]),
        np.median(pop_obj[:, 0]), 
        np.max(pop_obj[:, 0])))
    best_index = np.argmin(pop_obj[:, 0])
    best_genome = genome_convert(X[best_index, :], Config().layers_indexes)
    print('Best genome: {}'.format(best_genome))
    print('Result = {:.3f}, params number = {:.2e}'
        .format(1-pop_obj[best_index, 0], pop_obj[best_index, 1]))
    print('\n')


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

        X = X.astype(np.bool)
        _X = np.full(X.shape, np.inf)

        M = np.random.random(X.shape)
        flip, no_flip = M < self.prob, M >= self.prob

        _X[flip] = np.logical_not(X[flip])
        _X[no_flip] = X[no_flip]

        _X = _X.astype(np.int)

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

