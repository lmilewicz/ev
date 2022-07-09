from time import time
import numpy as np
from statistics import mean
import codecs, json 
import os
import time
from datetime import datetime

from misc import genome_convert


def log_stats(algorithm):
    gen = algorithm.n_gen
    pop_obj = algorithm.pop.get('F')
    X = algorithm.pop.get('X')
    config = algorithm.problem.config

    best_index = np.argmin(pop_obj[:, 0])
    best_genome = genome_convert(X[best_index, :], config.layers_indexes)

    if config.verbose:
        print('Gen = {}'.format(gen))
        print('Population error: best = {:.3f}, mean = {:.3f}, median = {:.3f}, worst = {:.3f}'.format(
            np.min(pop_obj[:, 0]), 
            np.mean(pop_obj[:, 0]),
            np.median(pop_obj[:, 0]), 
            np.max(pop_obj[:, 0])))

        print('Best genome: {}'.format(best_genome))
        print('Result = {:.3f}, params number = {:.2e}'
            .format(1-pop_obj[best_index, 0], pop_obj[best_index, 1]))

        print('Timestats: blueprint_time: '
            +str(round(mean(config.blueprint_time), 2))
            +'. fit_time: '
            +str(round(mean(config.fit_time), 2))
            +'. performance_time: '
            +str(round(mean(config.performance_time), 2)))
    else:
        print('Gen {}, error {:.3f}, params {:.2e}, time {:.2f}, {:.2f}, {:.2f}, genome {}'.format(
            gen,
            np.min(pop_obj[:, 0]),
            pop_obj[best_index, 1],
            round(mean(config.fit_time),2),
            round(mean(config.performance_time),2),
            round((time.time()-config.start_time), 2),
            best_genome))


def save_model(algorithm):
    gen = algorithm.n_gen
    X = algorithm.pop.get('X')
    config = algorithm.problem.config

    path_file = "model_json/"+str(config.time)
    if not os.path.exists(path_file):
        os.mkdir(path_file)

    genomes_file_path = path_file+"/genomes_gen_"+str(gen)+".json"
    bestmodel_file_path = path_file+"/bestmodel_gen_"+str(gen)+".json"

    if os.path.isfile(genomes_file_path) is False: genomes_file = []
    else:
        with open(genomes_file_path) as fp: genomes_file = json.load(fp)

    genomes_file.append(X.tolist())
    json.dump(genomes_file, codecs.open(genomes_file_path, 'w', encoding='utf-8'), 
            separators=(',', ':'), 
            sort_keys=True, 
            indent=4)

    if os.path.isfile(bestmodel_file_path) is False: bestmodel_file = []
    else:
        with open(bestmodel_file_path) as fp2: bestmodel_file = json.load(fp2)
    if config.best_model is not None: bestmodel_file.append(config.best_model.to_json())

    json.dump(bestmodel_file, codecs.open(bestmodel_file_path, 'w', encoding='utf-8'))

    if not algorithm.problem.config.log_stats: print('Generation = {} saved!'.format(gen))



def load_saved_state(time_str="", gen=0):
    
    if time_str == "":
        last_time = datetime.fromtimestamp(0)
        for _, dirs, _ in os.walk("model_json"):
            for dir in dirs:
                date_time_obj = datetime.strptime(dir, '%Y%m%d_%H%M%S')
                if date_time_obj > last_time:
                    time_str = dir
                    last_time = date_time_obj

    if gen == 0:
        for _, _, files in os.walk("model_json/"+time_str):
            for file in files: 
                if file[0] == "g":
                    number = ""
                    for m in file:
                        if m.isdigit(): number = number + m
                    if int(number) > gen: 
                        gen = int(number)
                        file_str = file
    print(time_str, file_str)

        # for name in files:
        #     if name.endswith(("lib", ".so")):
        #         os.path.join(root, name)


