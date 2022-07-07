import numpy as np
from statistics import mean
import codecs, json 
from os import path


from misc import genome_convert


def log_stats(algorithm):
    gen = algorithm.n_gen
    pop_obj = algorithm.pop.get('F')
    X = algorithm.pop.get('X')
    config = algorithm.problem.config


    print('Generation = {}'.format(gen))
    print('population error: best = {:.3f}, mean = {:.3f}, median = {:.3f}, worst = {:.3f}'.format(
        np.min(pop_obj[:, 0]), 
        np.mean(pop_obj[:, 0]),
        np.median(pop_obj[:, 0]), 
        np.max(pop_obj[:, 0])))
    best_index = np.argmin(pop_obj[:, 0])
    best_genome = genome_convert(X[best_index, :], config.layers_indexes)
    print('Best genome: {}'.format(best_genome))
    print('Result = {:.3f}, params number = {:.2e}'
        .format(1-pop_obj[best_index, 0], pop_obj[best_index, 1]))

    print('Timestats: blueprint_time: '
            +str(round(mean(config.blueprint_time), 2))
            +'. fit_time: '
            +str(round(mean(config.fit_time), 2))
            +'. performance_time: '
            +str(round(mean(config.performance_time), 2)))


def save_model(algorithm):
    gen = algorithm.n_gen
    X = algorithm.pop.get('X')
    config = algorithm.problem.config
    
    genomes_file_path = "model_json/genomes_"+str(config.time)+".json"
    if path.isfile(genomes_file_path) is False: genomes_file = []
    else:
        with open(genomes_file_path) as fp: genomes_file = json.load(fp)
    genomes_file.append(X.tolist())
    json.dump(genomes_file, codecs.open(genomes_file_path, 'w', encoding='utf-8'), 
            separators=(',', ':'), 
            sort_keys=True, 
            indent=4)

    bestmodel_file_path = "model_json/bestmodel_"+str(config.time)+".json"
    if path.isfile(bestmodel_file_path) is False: bestmodel_file = []
    else:
        with open(bestmodel_file_path) as fp2: bestmodel_file = json.load(fp2)
    if config.best_model is not None: bestmodel_file.append(config.best_model.to_json())

    json.dump(bestmodel_file, codecs.open(bestmodel_file_path, 'w', encoding='utf-8'))

    if not algorithm.problem.config.log_stats: print('Generation = {} saved!'.format(gen))
