import numpy as np

from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.model.mutation import Mutation

import tensorflow as tf

from blueprint import Blueprint, DenseLayer

def do_every_generations(algorithm):
    gen = algorithm.n_gen
    pop_obj = algorithm.pop.get("F")

    print("generation = {}".format(gen))
    print("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    
class EVProblem(Problem):
    def __init__(self, n_var, n_obj, n_constr, xl, xu, layers):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = xl
        self.xu = xu
        self.layers = layers

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)

        n_epochs = 5
        _batch_size = 512
        mnist_dataset = tf.keras.datasets.mnist.load_data()
        (train_images, train_labels), (test_images, test_labels) = mnist_dataset
        train_images, test_images = train_images / 255.0, test_images / 255.0

        for i in range(x.shape[0]):
            blueprint_object = Blueprint(genome = x, layers = self.layers, input_shape = (28, 28, 1), layerType = DenseLayer)
            model = blueprint_object.get_model()
            
            model.fit(x=train_images,
                        y=train_labels,
                        epochs=n_epochs,
                        use_multiprocessing=True,
                        batch_size=_batch_size,
                        verbose=0)

            performance = model.evaluate(test_images, test_labels, verbose=0)

            objs[i, 0] = 1 - performance

        out["F"] = objs

class MySampling(Sampling):
    def __init__(self) -> None:
        super().__init__()
        
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(np.int)


class MyMutation(Mutation):

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

        return _X.astype(np.int)

