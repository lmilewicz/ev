
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from datetime import datetime
import time

from data_set import DataSet
import test


class Config():
    def __init__(self, argv=[]):
        ### Model settings      ###
        self.dataset = 'mnist' # 'cifar10'  'mnist' 'cifar10_corrupted' 'cifar100'
        self.enable_xgboost = False
        self.batch_size = 32

        self.ds_train, self.ds_test, ds_info = DataSet(self.dataset, self.batch_size)
        self.dataset_size = ds_info.splits['train'].num_examples
        self.input_shape = ds_info.features['image'].shape
        self.out_units = ds_info.features['label'].num_classes

        ### Test settings ###
        self.save_model = self.log_stats = self.save_graph_visualization = True
        self.verbose = False
        self.debug = False

        ### Global test values ###
        self.best_model = None

        ### Saved files ###
        self.global_dir = "model_json"
        self.genomes_path = "genomes_gen_"
        self.best_model_path = "bestmodel_gen_"
        self.algorithm_path = "algorithm_last_state"

        ### Load files ###
        [self.load_gen, self.load_genomes, self.load_best_model, self.load_time_str] = [0, None, None, ""]
        if len(argv)>1 and int(argv[1]) > 0:
            [self.load_gen, self.load_genomes, self.load_best_model, self.load_time_str] = test.load_saved_state(self)

        if self.load_time_str == "":
            now = datetime.now()
            self.time = now.strftime("%Y%m%d_%H%M%S")
        else:
            self.time = self.load_time_str
        self.path_dir = self.global_dir+"/"+str(self.time)
        

        ### Evolution settings  ###
        self.max_n_conv_layers = 5
        self.max_n_ann_layers = 5

        self.n_conv_layers = 3
        self.n_ann_layers = 3

        self.number_of_objectives = 2
        #self.pop_size = 10                ##################
        self.pop_size = 2               ##################
        self.n_constr = 0
        self.algorithm = 'NSGA2'
        if len(argv)>1 and int(argv[1]) > 0:
            self.n_gen = int(argv[1])
        else:
            self.n_gen = 5
        self.termination = ('n_gen', self.n_gen)


        ### Genome settings     ###
        self.max_n_conv_modules = 2
        self.max_n_ann_modules = 1

        self.n_conv_modules = 1
        self.n_ann_modules = 1

        self.max_n_modules = self.max_n_conv_modules+self.max_n_ann_modules
        self.n_modules = self.n_conv_modules+self.n_ann_modules

        self.conv_module_genome_len = int(self.max_n_conv_layers*(self.max_n_conv_layers-1)*0.5)
        self.ann_module_genome_len = int(self.max_n_ann_layers*(self.max_n_ann_layers-1)*0.5)

        self.conv_genome_len = self.conv_module_genome_len*self.max_n_conv_modules
        self.ann_genome_len = self.ann_module_genome_len*self.max_n_ann_modules
        self.topology_len = self.conv_genome_len + self.ann_genome_len
        self.genome_len = self.topology_len + self.max_n_modules*2 + 1

        ### ANN settings        ###
        learning_rate = 0.02
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.n_epochs = 10           ##################
        self.out_activation = 'softmax'

        self.units = 16             ## To optimize
        self.kernel_size = 3        ## To optimize
        self.activation = 'relu'    ## To optimize
        self.dtype = tf.float32

        self.activation_array = [tf.keras.activations.relu, 
                tf.keras.activations.selu,
                tf.keras.activations.swish]

        ### XGBoost settings    ###
        self.xgboost_params = {
            'max_depth':3,
            # 'eta':0.05,
            'objective':'multi:softprob',
            'num_class': self.out_units,
            # 'early_stopping_rounds':10,
            'eval_metric':'merror'
        }
        self.xgboost_n_round = 10

        ### Global values/settings ###
        self.kl_divergence_function = (lambda q, p, _:
            tfp.distributions.kl_divergence(q, p)
            / tf.cast(self.dataset_size, dtype=self.dtype))
        
        self.conv_layers_indexes = get_layers_indexes(self.max_n_conv_layers)
        self.ann_layers_indexes = get_layers_indexes(self.max_n_ann_layers)


        ### Time benchmarks ###
        self.blueprint_time = []
        self.fit_time = []
        self.performance_time = []
        self.start_time = time.time()


        log_dir = 'logs'
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def get_layers_indexes(n_layers):
    layers_indexes = np.zeros(n_layers, dtype=np.int)
    idx = 0
    for i in range(n_layers):
        layers_indexes[i] = idx
        idx = idx+i+1 
    return layers_indexes


