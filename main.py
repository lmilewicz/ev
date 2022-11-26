# Semantation fault debug:
# import sys
# sys.settrace

# To disable Tensorflow warnings:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# # To solve 'ran out of gpu memory' in TensorFlow
# import tensorflow as tf
# tf_config = tf.compat.v1.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=tf_config)

import sys
import math

from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.optimize import minimize

import evolution, evolution_operations
from config import Config
# import test

# To speed calcualtions:
# import numba as nb
# @nb.njit
# or @nb.jit e.g.--> @nb.jit(nb.f8[:,:](nb.f8[:,:], nb.f8[:,:]), forceobj=True)


def main(argv=[], dataset = 'mnist', output='any'):

    config_settings = {
        'n_gen': 30,
        'load_gen': 0,
        'dataset': dataset,
        'pop_size': 20,
        'n_epochs': 15,
        'number_of_objectives': 2,
        # 'dropout': 0.4
        'batch_size': 64,
        'output': output
        }
    if len(argv)>1 and int(argv[1]) > 0: config_settings['argv'] = int(argv[1])


    iterations = 1
    iteration_for_second_objective = 2

    for i in range(iterations):
        config = Config(config_settings)      

        problem = evolution.EVProblem(config)
        algorithm = NSGA2(pop_size=config.pop_size,
                    sampling=evolution_operations.SamplingFromSmall(),
                    mutation=evolution_operations.MutationFromSmall(),
                    eliminate_duplicates=True)
        res = minimize(problem,
                        algorithm, 
                        callback=evolution.do_every_generations,
                        termination=config.termination)

        config_settings['load_gen'] = config_settings['n_gen'] + config_settings['load_gen']
        config_settings['load_genomes'] = res.pop.get("X").tolist()
        config_settings['load_best_model'] = res.pop.get("X")[0].tolist()
        config_settings['load_time_str'] = config.time

        if i == iteration_for_second_objective-1: config_settings['number_of_objectives'] = 2

        config_settings['pop_size'] = math.ceil(config_settings['pop_size']*0.66)
        config_settings['n_epochs'] = math.ceil(config_settings['n_epochs']*1.33)
        config_settings['load_genomes'] = config_settings['load_genomes'][0:config_settings['pop_size']]

if __name__ == '__main__':
    dataset='cifar100'
    output='any'
    main(sys.argv, dataset=dataset, output=output)





# def main():
#     config = Config(sys.argv)
       
#     problem = evolution.EVProblem(config)

#     # algorithm = NSGA2(pop_size=config.pop_size,
#     #             sampling=evolution_operations.SamplingAll(),
#     #             mutation=evolution_operations.MutationAll(),
#     #             eliminate_duplicates=True)
#     algorithm = NSGA2(pop_size=config.pop_size,
#                 sampling=evolution_operations.SamplingFromSmall(),
#                 mutation=evolution_operations.MutationFromSmall(),
#                 eliminate_duplicates=True)

#     res = minimize(problem,
#                     algorithm, 
#                     callback=evolution.do_every_generations,
#                     termination=config.termination)


