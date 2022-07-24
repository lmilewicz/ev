from graphviz import Digraph


def get_graph(module):
    graph = {}

    graph[1] = []

    for i in range(len(module) - 1):
        graph[i + 2] = [j + 1 for j in range(len(module[i])) if module[i][j] == 1]

    return graph


def visualize_genome_2(genome):

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    g = Digraph(format="pdf", filename='genome.gv', node_attr=node_attr, graph_attr=dict(size="12,12"))
    g.node(str("input"), "Input")

    with g.subgraph(name='part_1') as p:
        p.attr(fillcolor="gray", label='', fontcolor="black", style="filled")

        for i in range(3):
            g.node('node'+str(i), 'nodeX'+str(i), fillcolor="lightblue", shape="circle")

    return g


def visualize_genome(genome, title=None, type="residual"):
    node_color = "lightblue"
    conv1x1_color = "white"
    sum_color = "green4"
    pool_color = "orange"
    phase_background_color = "gray"
    fc_color = "gray"

    node_shape = "circle"
    conv1x1_shape = "doublecircle"

    structure = []

    # Build node ids and names to make building graph easier.
    for i, gene in enumerate(genome):
        all_zeros = sum([sum(t) for t in gene[:-1]]) == 0

        if all_zeros:
            continue  # Skip everything is a gene is all zeros.

        prefix = "gene_" + str(i)
        phase = ("cluster_" + str(i + 1), "Phase " + str(i + 1))

        nodes = [(prefix + "_node_0", ' ')] \
            + [(prefix + "_node_" + str(j + 1), ' ') for j in range(len(gene) + 1)]

        pool = (prefix + "_pool", "Pooling")


        edges = []
        graph = get_graph(gene)
        print(gene)

        for sink, dependencies in graph.items():
            print(sink, dependencies)
            for source in dependencies:
                edges.append((nodes[source][0], nodes[sink][0]))

        structure.append(
            {
                "nodes": nodes,
                "edges": edges,
                "pool": pool,
                "phase": phase,
                "all_zeros": all_zeros,
                "graph": graph
            }
        )

    final_pool = structure[-1]["pool"]
    new_pool = (final_pool[0], "Avg. Pooling")
    structure[-1]["pool"] = new_pool

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(format="pdf", filename='genome.gv', node_attr=node_attr, graph_attr=dict(size="12,12"))
    dot.attr(rankdir="UD")

    if title:
        dot.attr(label=title+"\n\n")
        dot.attr(labelloc='t')

    dot.node(str("input"), "Input")

    for j, struct in enumerate(structure):
        nodes = struct['nodes']
        edges = struct['edges']
        phase = struct['phase']
        pool = struct['pool']
        graph = struct['graph']
        all_zeros = struct['all_zeros']

        # Add nodes.
        dot.node(nodes[0][0], nodes[0][1], fillcolor="white", shape="doublecircle")

        if j > 0:
            dot.edge(structure[j - 1]['pool'][0], nodes[0][0])

        if not all_zeros:
            with dot.subgraph(name=phase[0]) as p:
                p.attr(fillcolor=phase_background_color, label='', fontcolor="black", style="filled")

                for i in range(1, len(nodes) - 1):
                    if len(graph[i]) != 0:
                        p.node(nodes[i][0], nodes[i][1], fillcolor=node_color, shape=node_shape)

        dot.node(nodes[-1][0], nodes[-1][1], fillcolor=sum_color, shape=node_shape)
        dot.node(*pool, fillcolor=pool_color)

        # Add edges.
        for edge in edges:
            dot.edge(*edge)

        dot.edge(nodes[-1][0], pool[0])

    dot.edge("input", structure[0]['nodes'][0][0])

    dot.node("linear", "Linear", fillcolor=fc_color)
    dot.edge(structure[-1]['pool'][0], "linear")


    return dot



if __name__ == "__main__":
    genotype_name = "Genome"

    genome = [[[1], [1, 1], [1, 1, 1]],
              [[1], [1, 1], [1, 1, 1]],
              [[1], [1, 1], [1, 1, 1], [0]]]

    d = visualize_genome(genome)
    d.view()
