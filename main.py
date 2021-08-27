# To disable Tensorflow warnings:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize

import node
import evolution

### CONFIG:::
class Config():
    def __init__(self):
        self.max_layers = 5
        self.pop_size = 10
        self.n_obj = 10
        self.n_constr = 0
        self.termination = ('n_gen', 10)
        self.dataset = 'MNIST'
        self.input_shape = (28, 28, 1)
        self.n_epochs = 5
        self.batch_size = 512
        self.layerType = node.DenseLayer    #  node.DenseLayer or node.DenseFlipout
        self.learning_rate = 0.01
        self.units = 10
        self.activation = 'relu'
        self.out_units = 10
        self.out_activation = 'softmax'
        self.layers_indexes = np.zeros(self.max_layers, dtype=np.int)
        
        idx = 0
        for i in range(self.max_layers):
            self.layers_indexes[i] = idx
            idx = idx+i+1


config = Config()

def main():
    problem = evolution.EVProblem(config)

    algorithm = NSGA2(pop_size=config.pop_size,
                        sampling=evolution.SamplingAll(),
                        mutation=evolution.MutationAll(),
                        eliminate_duplicates=True)

    res = minimize(problem, algorithm, callback=evolution.do_every_generations, termination=config.termination)


if __name__ == "__main__":
    main()