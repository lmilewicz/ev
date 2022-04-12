import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf

from data_loader import DataLoader

'''
Issues comments:
- Segmentation fault: ulimit -s to see current value; increase it with ulimit -s <new_value>

'''


''' To-do list:
1)
- Add last bit on genome that signifies the output layer type   ### done
- Set up evaluation for:                                        ### done
    - probabilistic layer                                       ### done
    - XGboost                                                   ### done
- Add XGBoost as output layer                                   ### done
- Improve mutation/sampling with different output layers

Testing:
- save genomes/models/testing state after specified number of generations/time spend
- stop testing and save testing state with an input key
- enable testing from saved point
- set-up jupyter board for online improvements


2)
- Add probabilistic Dense and Conv layers
- Different Conv layers/conv filers
- How to use graph to show different layer types. Current graph is focused on connection 

3) Different modules connections? Non sequential modules connection?

'''

class Config():
    def __init__(self):
        self.n_layers = 3
        self.pop_size = 2           ##################
        self.n_constr = 0

        self.algorithm = 'NSGA2'
        self.n_gen = 2             ##################
        self.termination = ('n_gen', self.n_gen)

        self.learning_rate = 0.001
        self.n_epochs = 1           ##################
        self.out_activation = 'softmax'

        self.units = 16             ## To optimize
        self.kernel_size = 3        ## To optimize
        self.activation = 'relu'    ## To optimize
        self.dtype = tf.float32

        self.n_conv_modules = 1
        self.n_ann_modules = 1
        self.n_modules = self.n_conv_modules+self.n_ann_modules
        self.module_genome_len = int(self.n_layers*(self.n_layers-1)*0.5)
        self.genome_len = self.module_genome_len*self.n_modules + 1 # +1 for output bit

        self.layers_indexes = np.zeros(self.n_layers, dtype=np.int)
        idx = 0
        for i in range(self.n_layers):
            self.layers_indexes[i] = idx
            idx = idx+i+1

        self.dataset = 'mnist' # 'mnist' 'cifar10'
        self.batch_size = 128

        self.ds_train, self.ds_test, ds_info = DataLoader(self.dataset, self.batch_size)
        self.dataset_size = ds_info.splits['train'].num_examples
        self.input_shape = ds_info.features['image'].shape
        self.out_units = ds_info.features['label'].num_classes

        self.kl_divergence_function = (lambda q, p, _:
            tfp.distributions.kl_divergence(q, p)
            / tf.cast(self.dataset_size, dtype=self.dtype))

        self.debug = False

        self.blueprint_time = []
        self.fit_time = []
        self.performance_time = []


