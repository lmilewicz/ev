import node
import numpy as np

class Config():
    def __init__(self):
        self.max_layers = 8
        self.pop_size = 4
        self.n_obj = 4
        self.n_constr = 0
        self.termination = ('n_gen', 5)
        self.dataset = 'MNIST'
        self.input_shape = (28, 28)
        self.n_epochs = 10
        self.batch_size = 128
        self.layerType = node.DenseLayer    #  node.DenseLayer or node.DenseFlipout ## To optimize
        self.learning_rate = 0.001  ## To optimize
        self.units = 16             ## To optimize
        self.activation = 'relu'
        self.out_units = 10
        self.out_activation = 'softmax'
        self.layers_indexes = np.zeros(self.max_layers, dtype=np.int)

        self.algorithm = 'NSGA2' #
        
        idx = 0
        for i in range(self.max_layers):
            self.layers_indexes[i] = idx
            idx = idx+i+1