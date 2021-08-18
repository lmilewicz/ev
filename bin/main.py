import tfne
import tensorflow as tf

from tfne.encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprintNode
from tfne.encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprintConn

from tfne.deserialization.codeepneat.codeepneat_deserialization import deserialize_codeepneat_module

# enc = tfne.encodings.CoDeepNEATEncoding(dtype=dtype)
# pop = tfne.populations.CoDeepNEATPopulation()

# Get preprocessed information from blueprint required for TF model creation
node_species = dict()
node_dependencies = dict()
graph_topology = list()

# node_dependencies[1] = set()
# dependencyless = set()
# for node, dep in node_dependencies.items():
#     if len(dep) == 0:
#         dependencyless.add(node)

# graph_topology.append(dependencyless)

# print(node_dependencies)
# print(graph_topology)

blueprint_graph = dict()
blueprint_graph[1] = CoDeepNEATBlueprintNode(gene_id=1, node=1, species=1)
blueprint_graph[2] = CoDeepNEATBlueprintNode(gene_id=1, node=2, species=1)
blueprint_graph[3] = CoDeepNEATBlueprintConn(gene_id=1, conn_start=1, conn_end=2, enabled=True)
blueprint_graph[4] = CoDeepNEATBlueprintNode(gene_id=1, node=3, species=1)
blueprint_graph[5] = CoDeepNEATBlueprintConn(gene_id=1, conn_start=1, conn_end=3, enabled=True)
blueprint_graph[6] = CoDeepNEATBlueprintConn(gene_id=1, conn_start=3, conn_end=2, enabled=True)


# blueprint_graph = [{'node': 1, 'species': 1}, 
#                     {'node': 2, 'species': 1}, 
#                     {'conn_start': 1, 'conn_end': 2, 'enabled': True}]
# "4": {
#     "node": 3,
#     "species": 1
# },
# "5": {
#     "conn_start": 1,
#     "conn_end": 3,
#     "enabled": true
# },
# "6": {
#     "conn_start": 3,
#     "conn_end": 2,
#     "enabled": true
# },
# Create set of species (self.species, set), assignment of nodes to their species (self.node_species, dict) as
# well as the assignment of nodes to the nodes they depend upon (self.node_dependencies, dict)
for gene in blueprint_graph.values():
    if isinstance(gene, CoDeepNEATBlueprintNode):
        node_species[gene.node] = gene.species
        # species.add(gene.species)
    elif gene.enabled:  # and isinstance(gene, CoDeepNEATBlueprintConn):
        # Only consider a connection for the processing if it is enabled
        if gene.conn_end in node_dependencies:
            node_dependencies[gene.conn_end].add(gene.conn_start)
        else:
            node_dependencies[gene.conn_end] = {gene.conn_start}

# Remove the 'None' species assigned to Input node
# species.remove(None)

# Topologically sort the graph and save into self.graph_topology as a list of sets of levels, with the first
# set being the layer dependent on nothing and the following sets depending on the values of the preceding sets
node_deps = node_dependencies.copy()

# print(node_deps)

node_deps[1] = set()  # Add Input node 1 to node dependencies as dependent on nothing
# print(node_deps)

while True:
    # find all nodes in graph having no dependencies in current iteration
    dependencyless = set()
    for node, dep in node_deps.items():
        if len(dep) == 0:
            dependencyless.add(node)

    print(node_deps, dependencyless, graph_topology)
    if not dependencyless:
        # If node_dependencies not empty, though no dependencyless node was found then a CircularDependencyError
        # occured

        # if node_deps:
        #     print("Circular Dependency Error when sorting the topology")

        # Otherwise if no dependencyless nodes exist anymore and node_deps is empty, exit dependency loop
        # regularly
        break

    # Add dependencyless nodes of current generation to list
    graph_topology.append(dependencyless)

    # remove keys with empty dependencies and remove all nodes that are considered dependencyless from the
    # dependencies of other nodes in order to create next iteration
    for node in dependencyless:
        del node_deps[node]
    for node, dep in node_deps.items():
        node_deps[node] = dep - dependencyless


# Create the actual Tensorflow model through the functional keras API, starting with the inputs object and
# saving the output of each layer in a dict that associates it with the node and serves for a later reference
# in the functional style of model creation.
inputs = tf.keras.Input(shape=(4, 4, 2), dtype=float)
node_outputs = {1: inputs}

bp_assigned_modules = dict()


bp_assigned_modules_dict = dict()
bp_assigned_modules_dict[1] = {
        'module_type': 'Conv2DMaxPool2DDropout',
        "module_id": 2,
        "parent_mutation": {
            "parent_id": None,
            "mutation": "init"
        },
        "merge_method": {
            "class_name": "Concatenate",
            "config": {
                "axis": -1,
                "dtype": "float32"
            }
        },
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "padding": "valid",
        "activation": "elu",
        "kernel_init": "glorot_uniform",
        "bias_init": "zeros",
        "max_pool_flag": False,
        "max_pool_size": 2,
        "dropout_flag": False,
        "dropout_rate": 0.2
}
# print(bp_assigned_modules_dict)
for spec, assigned_mod in bp_assigned_modules_dict.items():
    bp_assigned_modules[int(spec)] = deserialize_codeepneat_module(assigned_mod,
                                                                float,
                                                                module_config_params=None)

print(node_species)

i=0
for topology_level in graph_topology[1:]:
    for node in topology_level:
        # Determine the specific module of the current node and the required nodes the current node depends upon
        # print(node, node_species[node])
        
        current_node_module = bp_assigned_modules[node_species[node]]
        current_node_dependencies = tuple(node_dependencies[node])

        # As the current node only has 1 input, set this input node as the input for the current node
        node_input = node_outputs[current_node_dependencies[0]]

        # Create the sequential layers of the module and pipe the just created input through this node/module
        node_layers = current_node_module.create_module_layers()
        node_out = node_input
        for layer in node_layers:
            node_out = layer(node_out)

        # Register the final output of the sequential module layers as the output of the current node
        node_outputs[node] = node_out

        print('AAAAAAAAAA', i, node, topology_level)
        i=i+1

output_layers = [{'class_name': 'Flatten', 'config': {}},
                {'class_name': 'Dense', 'config': {'units': 10, 'activation': 'softmax'}}]

# Create the static output layers set by config and Pipe the results of the dynamic graph of modules through
# them. The dynamic graph always has the output node 2, which is therefore the input to the output layers.
deserialized_output_layers = [tf.keras.layers.deserialize(layer_config) for layer_config in output_layers]
# outputs = node_outputs[2]
outputs = node_outputs[2]


for out_layer in deserialized_output_layers:
    outputs = out_layer(outputs)

# Create the complete keras Model through the functional API by identifying the inputs and output layers
model = tf.keras.Model(inputs, outputs)

model.summary()

