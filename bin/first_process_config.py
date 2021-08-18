import tfne
from tfne.helper_functions import read_option_from_config

config = tfne.parse_configuration('./config-file.cfg')

# def _process_config
bp_pop_size = read_option_from_config(config, 'POPULATION', 'bp_pop_size')
mod_pop_size = read_option_from_config(config, 'POPULATION', 'mod_pop_size')
genomes_per_bp = read_option_from_config(config, 'POPULATION', 'genomes_per_bp')

# Read and process the config values that concern the genome creation for CoDeepNEAT
dtype = read_option_from_config(config, 'GENOME', 'dtype')
available_modules = read_option_from_config(config, 'GENOME', 'available_modules')
available_optimizers = read_option_from_config(config, 'GENOME', 'available_optimizers')
output_layers = read_option_from_config(config, 'GENOME', 'output_layers')

# Adjust output_layers config to include the configured datatype
for out_layer in output_layers:
    out_layer['config']['dtype'] = dtype

# Read and process the config values that concern the parameter range of the modules for CoDeepNEAT
available_mod_params = dict()
for available_mod in available_modules:
    # Determine a dict of all supplied configuration values as literal evals
    config_section_str = 'MODULE_' + available_mod.upper()
    if not config.has_section(config_section_str):
        raise RuntimeError(f"Module '{available_mod}' marked as available in config does not have an "
                            f"associated config section defining its parameters")
    mod_section_params = dict()
    for mod_param in config.options(config_section_str):
        mod_section_params[mod_param] = read_option_from_config(config, config_section_str, mod_param)

    # Assign that dict of all available parameters for the module to the instance variable
    available_mod_params[available_mod] = mod_section_params

# Read and process the config values that concern the parameter range of the optimizers for CoDeepNEAT
available_opt_params = dict()
for available_opt in available_optimizers:
    # Determine a dict of all supplied configuration values as literal evals
    config_section_str = 'OPTIMIZER_' + available_opt.upper()
    if not config.has_section(config_section_str):
        raise RuntimeError(f"Optimizer '{available_opt}' marked as available in config does not have an "
                            f"associated config section defining its parameters")
    opt_section_params = dict()
    for opt_param in config.options(config_section_str):
        opt_section_params[opt_param] = read_option_from_config(config, config_section_str, opt_param)

    # Assign that dict of all available parameters for the optimizers to the instance variable
    available_opt_params[available_opt] = opt_section_params


# ne_algorithm = tfne.algorithms.CoDeepNEAT(config)
# environment = tfne.environments.CIFAR10Environment(weight_training=True, config=config, verbosity=0)

# _process_config()
# _sanity_check_config()

# # Declare variables of environment shapes to which the created genomes have to adhere to
# input_shape = None
# output_shape = None

# # If an initial state of the evolution was supplied, load and recreate this state for the algorithm as well as
# # its dependencies
# if initial_state_file_path is not None:
#     # Load the backed up state for the algorithm from file
#     with open(initial_state_file_path) as saved_state_file:
#         saved_state = json.load(saved_state_file)

#     # Initialize and register an associated CoDeepNEAT encoding and population outfitted with the saved state
#     enc = tfne.deserialization.load_encoding(serialized_encoding=saved_state['encoding'],
#                                                     dtype=dtype)
#     pop = tfne.deserialization.load_population(serialized_population=saved_state['population'],
#                                                     dtype=dtype,
#                                                     module_config_params=available_mod_params)
# else:
#     # Initialize and register a blank associated CoDeepNEAT encoding and population
#     enc = tfne.encodings.CoDeepNEATEncoding(dtype=dtype)
#     pop = tfne.populations.CoDeepNEATPopulation()