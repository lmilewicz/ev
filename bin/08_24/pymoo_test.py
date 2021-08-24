import numpy as np
# from search import train_search
# from search import micro_encoding
# from search import macro_encoding
# from search import nsganet as engine

from pymop.problem import Problem
from pymoo.optimize import minimize

import tensorflow as tf
from nsga import NSGA2

from blueprint import Blueprint, DenseLayer

pop_hist = []  # keep track of every evaluated architecture

def blueprint_convert(layers, x):
    blueprint_graph = []
    start = end = 0
    for i in range(layers-1):
        start = end
        end = start+i+1
        blueprint_graph.append(x[start:end])

    return blueprint_graph

# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_var, n_obj, n_constr, lb, ub, layers):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self.layers = layers

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        n_epochs = 5
        _batch_size = 512
        mnist_dataset = tf.keras.datasets.mnist.load_data()
        (train_images, train_labels), (test_images, test_labels) = mnist_dataset
        train_images, test_images = train_images / 255.0, test_images / 255.0

        for i in range(x.shape[0]):
            genome = blueprint_convert(self.layers, x[i, :])
            # blueprint_graph = [[1], [0, 1], [1, 1, 1]]
            blueprint_object = Blueprint(blueprint_graph = genome, input_shape = (28, 28, 1), layerType = DenseLayer)
            model = blueprint_object.get_model()
            # genome = macro_encoding.convert(x[i, :])
            # genome = blueprint_graph
            
            model.fit(x=train_images,
                        y=train_labels,
                        epochs=n_epochs,
                        use_multiprocessing=True,
                        batch_size=_batch_size,
                        verbose=0)

            # performance = train_search.main(genome=genome,
            #                                 search_space=self._search_space,
            #                                 init_channels=self._init_channels,
            #                                 layers=self._layers, cutout=False,
            #                                 epochs=self._epochs,
            #                                 expr_root=self._save_dir)
            
            performance = model.evaluate(test_images, test_labels, verbose=0)

            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 1 - performance#performance['valid_acc']
            # objs[i, 1] = performance['flops']

            # self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # # this function will be call every generation
    # # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # # report generation info to files
    # logging.info("generation = {}".format(gen))
    # logging.info("population error: best = {}, mean = {}, "
    #              "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
    #                                               np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    # logging.info("population complexity: best = {}, mean = {}, "
    #              "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
    #                                               np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))
    print("generation = {}".format(gen))
    print("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    # print("population complexity: best = {}, mean = {}, "
    #              "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
    #                                               np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))



from pymoo.visualization.scatter import Scatter

def main():
    np.random.seed(0)

    layers = 5
    max_connections = int((layers-1)*layers/2) # E.g. genome for 4 layers  - all connected: [1], [1, 1], [1, 1, 1]
    
    lb = np.zeros(max_connections)
    ub = np.ones(max_connections)

    problem = NAS(n_var=max_connections, n_obj=1, n_constr=0, lb=lb, ub=ub, layers=layers)

    _pop_size = 10
    # configure the nsga-net method
    method = NSGA2(pop_size=_pop_size,
                            n_offsprings=_pop_size,
                            eliminate_duplicates=True)

    res = minimize(problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', 10))


    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()
    return


if __name__ == "__main__":
    main()