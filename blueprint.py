from gc import callbacks
import tensorflow as tf
import tensorflow_probability as tfp

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# from xgboost import XGBClassifier
import xgboost as xgb

import pandas as pd 
import time
import numpy as np

from module import Module
import node


class Blueprint():
    def __init__(self, genome, config) -> tf.keras.Model:
        self.model = None
        self.modules_list = []
        self.config = config
        self.genome = genome
        self.xgboost_input_layer = None

        if config.debug: print('Blueprint:', genome)
        
        time1 = time.time()

        tf.keras.backend.clear_session()
        
        input = tf.keras.Input(shape=config.input_shape, dtype=config.dtype)
        layer_type = node.Convolution2D

        last_layer = input
        
        module_genome_len = config.conv_module_genome_len
        layers_indexes=config.conv_layers_indexes

        for i in range(config.n_modules):
            if i == config.n_conv_modules:
                last_layer = node.MaxPool2D()(last_layer)
                last_layer = tf.keras.layers.Flatten()(last_layer)
                layer_type = node.DenseLayer
                module_genome_len = config.ann_module_genome_len
                layers_indexes=config.ann_layers_indexes


            genome_module = genome[module_genome_len*i:module_genome_len*(i+1)]
            if sum(genome_module) == 0: continue

            module = Module(genome_module, config, layer_type, layers_indexes, input_layer = last_layer)

            last_layer = module.get_module()

        if genome[-1] == 0:
            output = tf.keras.layers.Dense(
                        units=config.out_units, 
                        name='out_dense_layer',
                        activation=config.out_activation,
                        dtype=config.dtype)(last_layer)
        elif genome[-1] == 1:
            output = tfp.layers.DenseFlipout(
                        units=config.out_units,
                        name='out_prob_layer',
                        kernel_divergence_fn=config.kl_divergence_function,
                        activation=config.out_activation,
                        dtype=config.dtype)(last_layer)
        elif genome[-1] == 2:
            #     TO DO --> XGboost
            output = tf.keras.layers.Dense(
                        units=config.out_units, 
                        name='out_dense_layer',
                        activation=config.out_activation,
                        dtype=config.dtype)(last_layer)
            
            self.xgboost_input_layer = tf.keras.Model(inputs=input, outputs=last_layer)

        else:
            raise ValueError('In Blueprint: undefined output: '+str(genome[-1]))

        self.model = tf.keras.Model(input, output)

        if config.debug: print(self.model.summary())

        # self.model.compile(
        #     optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        config.blueprint_time.append(time.time()-time1)


    def get_model(self) -> tf.keras.Model:
        return self.model


    def evaluate_model(self):
        time1 = time.time()

        self.model.fit(self.config.ds_train,
                    epochs=self.config.n_epochs,
                    use_multiprocessing=True,
                    batch_size=self.config.batch_size,
                    verbose=0,)
                    #callbacks=[self.config.tensorboard_callback])

        time2 = time.time()
        self.config.fit_time.append(time2-time1)

        if self.genome[-1] == 0:
            performance = dense_performance(model=self.model, validation_data=self.config.ds_test)

        elif self.genome[-1] == 1:
            performance = prob_performance(model=self.model, validation_data=self.config.ds_test)

        elif self.genome[-1] == 2:
            performance = xgb_performance(
                            config=self.config, 
                            xgboost_input_layer=self.xgboost_input_layer)

        self.config.performance_time.append(time.time()-time2)

        print("Performance for model: {:.3f}".format(1-performance))
        return performance


def dense_performance(model, validation_data):
    evaluation = model.evaluate(validation_data, verbose=0)

    return evaluation[-1]


def prob_performance(model, validation_data):
    evaluation = [model.evaluate(validation_data, verbose=0)[-1] for _ in range(10)]

    return np.mean(evaluation)


def xgb_performance(config, xgboost_input_layer):
    # XGBoost output
    xgboost_train_input = xgboost_input_layer.predict(config.ds_train) 
    xgboost_train_input = pd.DataFrame(data=xgboost_train_input)
    train_labels_batched = list(map(lambda x: x[1], config.ds_train))
    train_labels = np.array([])
    for batch in train_labels_batched:
        train_labels = np.concatenate((train_labels, batch.numpy()), axis=0)

    xgboost_test_input = xgboost_input_layer.predict(config.ds_test) 
    xgboost_test_input = pd.DataFrame(data=xgboost_test_input)
    test_labels_batched = list(map(lambda x: x[1], config.ds_test))
    test_labels = np.array([])
    for batch in test_labels_batched:
        test_labels = np.concatenate((test_labels, batch.numpy()), axis=0)

    dtrain = xgb.DMatrix(xgboost_train_input, label=train_labels)
    dtest = xgb.DMatrix(xgboost_test_input, label=test_labels)

    watchlist = [(dtrain, 'train'),(dtest, 'eval')]

    results = {}

    model = xgb.train(config.xgboost_params, dtrain, config.xgboost_n_round, watchlist, verbose_eval=False, evals_result=results)
    model.__del__()

    return 1-results['eval']['merror'][-1]



