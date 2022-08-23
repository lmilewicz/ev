from graphviz import Digraph

from misc import get_graph, get_best_genome
from config import Config


# def visualize_genome(genome, title=None):
#     dot = Digraph(format='pdf', filename='genome2.gv', node_attr={'style':'filled'}, graph_attr={'rankdir':'LR', 'label':title})

def visualize_genome(algorithm):
    config = algorithm.problem.config
    genome = get_best_genome(algorithm, config)
    dot = visualize_genome_main(genome, config, gen = algorithm.n_gen)
    return dot


def visualize_genome_main(genome, config, gen=0):
    dot = Digraph(
        format='pdf', 
        filename=config.path_dir+'/'+'genome_gen_'+str(gen), 
        node_attr={'style':'filled'}, 
        graph_attr={'rankdir':'LR', 'label':'Best Genome'})

    input_str = 'input'
    output_str = 'input'
    dot.node(input_str, 'Input')
    set_input_connection = True

    for i, gene in enumerate(genome):
        if all(sum(layer) == 0 for layer in gene): continue
        if set_input_connection: 
            dot.edge(input_str, 'module_'+str(i)+'_node_1')
            set_input_connection = False

        graph = get_graph(gene)

        for output, inputs in graph.items():
            if output == 'concat':
                output_str = add_graph_connection(output_str, i, output, inputs, dot, concat=True)
            else:
                with dot.subgraph(name='cluster_'+str(i)) as c:
                    output_str = add_graph_connection(output_str, i, output, inputs, dot=c)

        if i < len(genome)-1: dot.edge(output_str, 'module_'+str(i+1)+'_node_1')

    dot.node('output', 'Output')
    dot.edge(output_str, 'output')
<<<<<<< HEAD

    #dot.view()
=======
    dot.render()
>>>>>>> 8df8591e2819ef3d2163f92dcf17553e33e09001

    return dot


def add_graph_connection(output_str, i, output, inputs, dot, concat=False):
    output_str = 'module_'+str(i)+'_node_'+str(output)
    dot.attr(color='black', label='')
    for input in inputs:
        input_str = 'module_'+str(i)+'_node_'+str(input)
        if concat: dot.node(output_str, fillcolor='green', label = str(output))
        else: dot.node(output_str, fillcolor='lightblue', label = str(output), shape='circle')
        dot.node(input_str, fillcolor='lightblue', label = str(input), shape='circle')
        dot.edge(input_str, output_str)

    return output_str



if __name__ == "__main__":
    genome = [[[1], [0, 0], [1, 1, 0]], [[0], [1, 0], [1, 0, 1]], [[1], [1, 1], [1, 1, 1], [0]]]

<<<<<<< HEAD
    visualize_genome(genome, 'Best Genome')
=======
    dot = visualize_genome_main(genome, Config())
    dot.view()
>>>>>>> 8df8591e2819ef3d2163f92dcf17553e33e09001
