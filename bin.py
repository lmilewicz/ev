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


########################
#evolution_operations.py
########################

# class SamplingAll(Sampling):
#     def __init__(self) -> None:
#         super().__init__()
        
#     def _do(self, problem, n_samples, **kwargs):
#         if problem.config.load_genomes != None:
#             _X = np.array(problem.config.load_genomes.copy())
#         else:
#             _X = np.random.random((n_samples, problem.n_var))
#             _X = (_X > 0.5).astype(np.int)

#         return RemoveDisconnectedLayers(_X, problem.config).return_new_X()


# class MutationAll(Mutation):
#     def __init__(self, prob=None):
#         super().__init__()
#         self.prob = prob
    
#     def _do(self, problem, X, **kwargs):
#         if self.prob is None:
#             self.prob = 1.0 / problem.n_var

#         if problem.config.enable_xgboost:
#             _X = xgboost_mutation(X, self.prob)
#         else:
#             _X = no_xgboost_mutation(X, self.prob)

#         return RemoveDisconnectedLayers(_X, problem.config).return_new_X()



# def random_mutation(dice, prob=0.1):
#     return dice > 1-prob


# def xgboost_mutation(X, R, prob):
#     X_without_output = X[:,:-1].copy().astype(bool)
#     X_output = X[:,-1].copy()

#     _X = np.full(X_without_output.shape, np.inf)

#     M = np.random.random(X_without_output.shape)
#     flip, no_flip = M < prob and R, M >= prob or R

#     _X[flip] = np.logical_not(X_without_output[flip])
#     _X[no_flip] = X_without_output[no_flip]
#     _X[~R] = False

#     _X = _X.astype(np.int)

#     ### Output layer mutation::: ###
#     X_output = X_output.reshape(len(X_output),1)
#     M = np.random.random(X_output.shape)
#     flip, no_flip = M < prob/2, M >= prob/2

#     X_output[flip] = np.mod(X_output[flip]+1, 3)
#     X_output[no_flip] = X_output[no_flip]
#     #################################

#     _X = np.append(_X, X_output, axis=1)

#     return _X



# def no_xgboost_mutation(X, R, prob):
#     X = X.astype(np.bool)
#     _X = np.full(X.shape, np.inf)

#     M = np.random.random(X.shape)
#     flip, no_flip = M < prob, M >= prob

#     _X[flip] = np.logical_not(X[flip])
#     _X[no_flip] = X[no_flip]
#     _X[~R] = False

#     _X = _X.astype(np.int)

#     return _X


# class OldMutationFromSmall(Mutation):
#     def __init__(self, prob=None):
#         super().__init__()
#         self.prob = prob

#     def _do(self, problem, X, **kwargs):
#         if self.prob is None:
#             self.prob = 1.0 / problem.n_var
        
#         _X = np.full(X.shape, 0)
#         config = problem.config

#         R = old_update_range(X.shape, config)

#         if self.prob is None:
#             self.prob = 1.0 / problem.n_var

#         if problem.config.enable_xgboost:
#             _X = xgboost_mutation(X, R, self.prob)
#         else:
#             _X = no_xgboost_mutation(X, R, self.prob)

#         # return remove_disconnected_layers(_X, problem.config)
#         return RemoveDisconnectedLayers(_X, problem.config).return_new_X()


# def old_update_range(shape, config):
#     random = np.random.random(8)
#     #CNN
#     if(random_mutation(random[0], 0.02) and config.n_conv_modules < config.max_n_conv_modules): 
#         config.n_conv_modules = config.n_conv_modules + 1
#     elif(random_mutation(random[1], 0.02) and config.n_conv_modules> 0): 
#         config.n_conv_modules = config.n_conv_modules - 1
#     if(random_mutation(random[2], 0.1) and config.n_conv_layers < config.max_n_conv_layers): 
#         config.n_conv_layers = config.n_conv_layers + 1
#     elif(random_mutation(random[3], 0.1) and config.n_conv_layers > 0): 
#         config.n_conv_layers = config.n_conv_layers - 1

#     #ANN
#     if(random_mutation(random[4], 0.02) and config.n_ann_modules < config.max_n_ann_modules): 
#         config.n_ann_modules = config.n_ann_modules + 1
#     elif(random_mutation(random[5], 0.02) and config.n_ann_modules > 0): 
#         config.n_ann_modules = config.n_ann_modules - 1
#     if(random_mutation(random[6], 0.1) and config.n_ann_layers < config.max_n_ann_layers): 
#         config.n_ann_layers = config.n_ann_layers + 1
#     elif(random_mutation(random[7], 0.1) and config.n_ann_layers > 0): 
#         config.n_ann_layers = config.n_ann_layers - 1

#     R = calculate_range(shape, config)

#     return R


# class SamplingFromSmall(Sampling):
#     # def __init__(self) -> None:
#     #     super().__init__()
        
#     # def _do(self, problem, n_samples, **kwargs):
#     #     if problem.config.load_genomes != None:
#     #         _X = np.array(problem.config.load_genomes.copy())
#     #     else:
#     #         _X = np.zeros((n_samples, problem.n_var))
#     #         _X[:,0] = np.random.random(n_samples)
#     #         _X = (_X > 0.5).astype(np.int)

#     #     return RemoveDisconnectedLayers(_X, problem.config).return_new_X()


# class MutationFromSmall(Mutation):
#     def __init__(self, prob=None):
#         super().__init__()
#         self.prob = prob

#     def _do(self, problem, X, **kwargs):
#         if self.prob is None:
#             self.prob = 1.0 / problem.n_var
        
#         _X = np.full(X.shape, 0)

#         for j in range(X.shape[0]):
#             break_loop = 0
#             for i in range(len(problem.config.layers_indexes)-1):
#                 start = problem.config.layers_indexes[i]
#                 end = problem.config.layers_indexes[i+1]
#                 X_layer = X[j, start:end].copy()
#                 _X_layer = X[j, start:end].copy()
#                 if(np.sum(X_layer))==0:
#                     _X_layer[end-start-1] = (1 if np.random.random() > self.prob else 0)
#                     break_loop = 1
#                 else:
#                     M = np.random.random(X_layer.shape)
#                     flip, no_flip = M < self.prob, M >= self.prob
#                     _X_layer[flip] = np.logical_not(X_layer[flip]).astype(np.int)
#                     _X_layer[no_flip] = X_layer[no_flip].astype(np.int)
#                     if(np.sum(X_layer)==0):
#                         break_loop=1
#                         _X[j, end:X.shape[1]] = np.zeros(X.shape[1]-end)
#                 _X[j, start:end] = _X_layer

#                 if(break_loop): break

#         return RemoveDisconnectedLayers(_X, problem.config).return_new_X()



###########################################
# main.py
###########################################

# from pymoo.algorithms.moo.unsga3 import UNSGA3
# from pymoo.algorithms.soo.nonconvex.isres import ISRES

# if config.algorithm == 'NSGA2': # https://pymoo.org/algorithms/moo/nsga2.html
#     algorithm = NSGA2(pop_size=config.pop_size,
#                         sampling=evolution_operations.SamplingAll(),
#                         mutation=evolution_operations.MutationAll(),
#                         eliminate_duplicates=True)
                        
# elif config.algorithm == 'UNSGA3': # https://pymoo.org/algorithms/moo/unsga3.html
#     ref_dirs = np.array([[0.0, 0.0]])
#     algorithm = UNSGA3(ref_dirs,
#                         sampling=evolution_operations.SamplingAll(),
#                         mutation=evolution_operations.MutationAll(),
#                         eliminate_duplicates=True)

# elif config.algorithm == 'ISRES': # https://pymoo.org/algorithms/soo/isres.html
#     algorithm = ISRES(pop_size=config.pop_size, 
#                         rule=1.0 / 7.0, 
#                         gamma=0.85, 
#                         alpha=0.2)
# else:
#     raise ValueError('Algorithm not chosen!')




#########################################################
# misc.py
#########################################################

# def get_params_dict(config, layer_type):
# 	params_dict = {'activation':	config.activation,
# 					'dtype':		tf.float32,
# 					'prob_layer':	False}
# 	if layer_type == node.DenseLayer:
# 		params_dict['units'] = config.units
# 	elif layer_type == node.Convolution2D:
# 		params_dict['kernel_size'] = config.kernel_size
# 	else:
# 		raise ValueError('In get_params_dict: layer_type with wrong type: '+str(type(layer_type)))
# 	return params_dict


# def old_remove_disconnected_layers(X, config):
#     _X = np.zeros(X.shape, dtype=np.int)

#     for i in range(X.shape[0]):
#         layers = np.zeros(config.n_layers*config.max_n_modules, dtype=np.int)

#         for j in range(config.max_n_modules):
#             genome_start = config.module_genome_len*j
#             genome_end = config.module_genome_len*(j+1)
            
#             genome_module = X[i, genome_start:genome_end]
#             layers[genome_start] = 1

#             genome_graph = module_convert(genome_module, layers_indexes=config.layers_indexes)
#             for idx, gene in enumerate(genome_graph, start=1):
#                 layer = 0
#                 gene_copy = gene.copy()
#                 if np.count_nonzero(gene) > 0:
#                     for j in np.nonzero(gene)[0]:
#                         if(layers[j] == 0): 
#                             gene_copy[j] = 0
#                         else:
#                             layer = 1
#                 layers[idx] = layer

#                 index_1 = config.layers_indexes[idx-1] + genome_start
#                 index_2 = config.layers_indexes[idx] + genome_start
#                 _X[i, index_1:index_2] = gene_copy

#     return _X


# def remove_disconnected_layers(X, config):
#     _X = np.zeros(X.shape, dtype=np.int)

#     for i in range(X.shape[0]):
#         layers = np.zeros(config.n_layers*config.max_n_modules, dtype=np.int)

#         for j in range(config.max_n_modules):
#             genome_start = config.module_genome_len*j
#             genome_end = config.module_genome_len*(j+1)
            
#             genome_module = X[i, genome_start:genome_end]
#             layers[genome_start] = 1

#             genome_graph = module_convert(genome_module, layers_indexes=config.layers_indexes)
#             for idx, gene in enumerate(genome_graph, start=1):
#                 layer = 0
#                 gene_copy = gene.copy()
#                 if np.count_nonzero(gene) > 0:
#                     for j in np.nonzero(gene)[0]:
#                         if(layers[j] == 0): 
#                             gene_copy[j] = 0
#                         else:
#                             layer = 1
#                 layers[idx] = layer

#                 index_1 = config.layers_indexes[idx-1] + genome_start
#                 index_2 = config.layers_indexes[idx] + genome_start
#                 _X[i, index_1:index_2] = gene_copy

#     return _X



# def get_flops(model):
#     ''' Returns GFLOPS value needed to process neural network model
#     '''
#     batch_size = 1
#     inputs = [tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs]
#     real_model = tf.function(model).get_concrete_function(inputs)
#     frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

#     run_meta = tf.compat.v1.RunMetadata()
#     opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#     opts['output'] = 'none'

#     flops = tf.compat.v1.profiler.profile(
#         graph=frozen_func.graph, run_meta=run_meta, cmd='scope', options=opts
#     )
#     return float(flops.total_float_ops) * 1e-9

