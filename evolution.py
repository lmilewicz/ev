import numpy as np
from datetime import datetime

from pymoo.core.problem import Problem

from blueprint import Blueprint
import misc
import test 
import visualization


class EVProblem(Problem):
    def __init__(self, config):
        # E.g. genome for 4 layers  - all connected: [1], [1, 1], [1, 1, 1] -> 6
        super().__init__(n_var=config.genome_len, n_obj=2, 
            n_constr=config.n_constr, type_var=np.int)

        self.config = config
        self.xl = np.zeros(config.genome_len)
        self.xu = np.ones(config.genome_len)
        for i in range(config.topology_len, config.genome_len):
            if i % 2: self.xu[i] = 2
            else: self.xu[i] = 5
        self.xu[-1] = 2

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        best_perf = 0

        # print("_evaluate")
        # print(x)

        best_model = None
        for i in range(x.shape[0]):
            
            blueprint_object = Blueprint(genome=x[i, :], config=self.config)
            model = blueprint_object.get_model_array()[0]
            performance = blueprint_object.evaluate_model()

            objs[i, 0] = 1 - performance
            if self.config.number_of_objectives > 1: objs[i, 1] = misc.get_params_number(model)
            else: objs[i, 1] = 999999

            if performance > best_perf:
                best_perf = performance
                best_model = model

            print(str(datetime.now().time().strftime("%H:%M:%S"))+": Error: {:.4f}, Complex: {:.2f}, genome {}"
                .format(objs[i, 0], objs[i, 1], x[i, :]))
            
        if self.config.debug:
            print('Best perf: '+str(round(best_perf, 4)))
            best_model.summary()

        self.config.best_model = best_model

        out['F'] = objs


        # print(objs)

def do_every_generations(algorithm):
    if algorithm.problem.config.log_stats: test.log_stats(algorithm)
    if algorithm.problem.config.save_model: test.save_model(algorithm)
    if algorithm.problem.config.save_graph_visualization: visualization.visualize_genome(algorithm)
