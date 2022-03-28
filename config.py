import tensorflow as tf
import numpy as np

from data_loader import DataLoader

class Config():
    def __init__(self):
        self.n_layers = 3
        self.pop_size = 4
        self.n_constr = 0        
        self.n_conv_modules = 1 
        self.n_ann_modules = 1       
        self.n_modules = self.n_conv_modules+self.n_ann_modules    

        self.module_genome_len = int(self.n_layers*(self.n_layers-1)*0.5)
        self.genome_len = self.module_genome_len*self.n_modules

        self.algorithm = 'NSGA2'

        self.n_epochs = 4
        self.n_gen = 5
        self.termination = ('n_gen', self.n_gen)
        self.dataset = 'MNIST'
        self.input_shape = (28, 28, 1)
        self.batch_size = 128

        self.learning_rate = 0.001          ## To optimize
        self.units = 16                     ## To optimize
        self.activation = 'relu'
        self.out_units = 10
        self.out_activation = 'softmax'

        self.layers_indexes = np.zeros(self.n_layers, dtype=np.int)
        idx = 0
        for i in range(self.n_layers):
            self.layers_indexes[i] = idx
            idx = idx+i+1

        self.ds_train, self.ds_test, ds_info = DataLoader(self.dataset, self.batch_size)
        self.dataset_size = ds_info.splits['train'].num_examples

        self.debug = False

        self.blueprint_time = []
        self.fit_time = []
        self.layer_count = 0
