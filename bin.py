

# def save_model(algorithm):

#     checkpoint_path = config.path_dir+"/"+time_str+"/"+config.algorithm_path+".npy"

#     algo_checkpoint, = np.load(checkpoint_path, allow_pickle=True).flatten()
#     algo_checkpoint.has_terminated = False



# def save_algorithm_state(config, algorithm): ### DOES NOT WORK!!! ### "Cannot convert a Tensor of dtype variant to a NumPy array."
#     algorithm_file_path = config.path_dir+"/"+str(config.time)+"/"+config.algorithm_path
#     np.save(algorithm_file_path, algorithm)


# def main();
#     if config.load_algo_checkpoint != None:
#         algorithm = config.load_algo_checkpoint
#         print(algorithm.n_gen)

#     else:

                    # copy_algorithm=False, --> for test.save_algorithm_state

    # test.save_algorithm_state(config, algorithm)

