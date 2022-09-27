import pandas as pd
import numpy as np
from statistics import mean
import codecs, json, os, time
from datetime import datetime
from tensorflow.keras.models import model_from_json

import misc


def log_stats(algorithm):
    gen = algorithm.n_gen+config.load_gen
    config = algorithm.problem.config

    pop_obj = algorithm.pop.get('F')
    X = algorithm.pop.get('X')
    best_index = np.argmin(pop_obj[:, 0])
    best_genome = misc.get_best_genome(algorithm, config)
    complexity = 999999
    if config.number_of_objectives > 1: complexity = pop_obj[best_index, 1]

    if config.verbose:
        print('Gen = {}'.format(gen))
        print('Population error: best = {:.3f}, mean = {:.3f}, median = {:.3f}, worst = {:.3f}'.format(
            np.min(pop_obj[:, 0]), 
            np.mean(pop_obj[:, 0]),
            np.median(pop_obj[:, 0]), 
            np.max(pop_obj[:, 0])))
        
        if not os.path.exists(config.path_dir): os.mkdir(config.path_dir)

        if config.number_of_objectives > 1: 
            data_to_save = np.asarray([gen, \
            np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]), np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0]),\
            np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]), np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])])
        else:
            data_to_save = np.asarray([gen, \
            np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]), np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])])
        with open(config.path_dir+'/final_data.csv', 'ab') as f:
            np.savetxt(f, data_to_save.reshape(1, data_to_save.shape[0]), fmt='%.6f', delimiter=',')

        print('Best genome: {}'.format(best_genome))
        print('Error = {:.3f}, params number = {:.2e}'.format(pop_obj[best_index, 0], complexity))
        print('Timestats: blueprint_time: '
            +str(round(mean(config.blueprint_time), 2))
            +'. fit_time: '
            +str(round(mean(config.fit_time), 2))
            +'. performance_time: '
            +str(round(mean(config.performance_time), 2)))
    else:
        
        print('Gen {}, error {:.3f}, params {:.2e}, time {:.2f}, {:.2f}, {:.2f}, genome {}'.format(
            gen,
            pop_obj[best_index, 0],
            complexity,
            round(mean(config.fit_time),2),
            round(mean(config.performance_time),2),
            round((time.time()-config.start_time), 2),
            best_genome))

    # for i, genome in enumerate(X):
    #     print(i, genome)


def save_model(algorithm):
    config = algorithm.problem.config

    X = algorithm.pop.get('X')
    gen = algorithm.n_gen+config.load_gen

    if not os.path.exists(config.path_dir): os.mkdir(config.path_dir)

    genomes_file_path = config.path_dir+"/"+config.genomes_path+str(gen)+".json"
    bestmodel_file_path = config.path_dir+"/"+config.best_model_path+str(gen)

    json.dump(X.tolist(), codecs.open(genomes_file_path, 'w', encoding='utf-8'), 
            separators=(',', ':'), 
            sort_keys=True, 
            indent=4)

    best_model_json = config.best_model.to_json()
    with open(bestmodel_file_path+".json", "w") as fp2:
        fp2.write(best_model_json)
    config.best_model.save_weights(bestmodel_file_path+".h5")

    if not config.log_stats: print('Generation = {} saved!'.format(gen))


def load_saved_state(config, time_str="", gen=0):
    if time_str == "":
        last_time = datetime.fromtimestamp(0)
        for _, dirs, _ in os.walk(config.global_dir):
            for dir in dirs:
                date_time_obj = datetime.strptime(dir, '%Y%m%d_%H%M%S')
                if date_time_obj > last_time:
                    time_str = dir
                    last_time = date_time_obj

    if time_str == "":
        now = datetime.now()
        config.time = now.strftime("%Y%m%d_%H%M%S")
    else:
        config.time = time_str
    config.path_dir = config.global_dir+"/"+str(config.time)

    if gen == 0:
        for _, _, files in os.walk(config.path_dir):
            for file in files: 
                if "gen" in file:
                    number = ""
                    for m in file:
                        if m.isdigit(): number = number + m
                        if m == ".": break
                    if int(number) > gen: gen = int(number)

    genomes_file_path = config.path_dir+"/"+config.genomes_path+str(gen)+".json"
    bestmodel_file_path = config.path_dir+"/"+config.best_model_path+str(gen)

    if os.path.isfile(genomes_file_path) is False: genomes = []
    else:
        with open(genomes_file_path) as fp: genomes = json.load(fp)

    json_file = open(bestmodel_file_path+".json", 'r')
    best_model_json = json_file.read()
    json_file.close()

    best_model = model_from_json(best_model_json)
    best_model.load_weights(bestmodel_file_path+".h5")

    return [gen, genomes, best_model, time_str]


