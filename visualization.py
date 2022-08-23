from graphviz import Digraph

from misc import get_graph


def visualize_genome(genome, title=None):
    dot = Digraph(format='pdf', filename='genome2.gv', node_attr={'style':'filled'}, graph_attr={'rankdir':'LR', 'label':title})

    input_str = 'input'
    output_str = 'input'
    dot.node(input_str, 'Input')
    dot.edge(input_str, 'module_0_node_1')

    for i, gene in enumerate(genome):
        graph = get_graph(gene)

        for output, inputs in graph.items():
            output_str = 'module_'+str(i)+'_node_'+str(output)

            with dot.subgraph(name='cluster_'+str(i)) as c:
                c.attr(color='black', label='')
                for input in inputs:
                    input_str = 'module_'+str(i)+'_node_'+str(input)
                    c.node(output_str, fillcolor='lightblue', label = str(output), shape='circle')
                    c.node(input_str, fillcolor='lightblue', label = str(input), shape='circle')
                    c.edge(input_str, output_str)

        if i < len(genome)-1: dot.edge(output_str, 'module_'+str(i+1)+'_node_1')


    dot.node('output', 'Output')
    dot.edge(output_str, 'output')

    #dot.view()

    return dot



if __name__ == "__main__":
    genome = [[[1], [0, 0], [1, 1, 0]],
              [[0], [1, 0], [1, 0, 1]],
              [[1], [1, 1], [1, 1, 1], [0]]]

    visualize_genome(genome, 'Best Genome')
