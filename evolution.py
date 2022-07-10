import numpy as np

from pymoo.core.problem import Problem

from blueprint import Blueprint
import misc
import test 

class EVProblem(Problem):
    def __init__(self, config):
        # E.g. genome for 4 layers  - all connected: [1], [1, 1], [1, 1, 1] -> 6
        super().__init__(n_var=config.genome_len, n_obj=config.pop_size, 
            n_constr=config.n_constr, type_var=np.int)

        self.config = config
        self.xl = np.zeros(config.genome_len)
        self.xu = np.ones(config.genome_len)

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        best_perf = 0

        best_model = None
        for i in range(x.shape[0]):
            # x[i, -1] = 2
            blueprint_object = Blueprint(genome=x[i, :], config=self.config)
            model = blueprint_object.get_model()
            performance = blueprint_object.evaluate_model()

            objs[i, 0] = 1 - performance
            objs[i, 1] = misc.get_params_number(model)

            if performance > best_perf:
                best_perf = performance
                best_model = model

        if self.config.debug:
            print('Best perf: '+str(round(best_perf, 4)))
            best_model.summary()

        self.config.best_model = best_model

        out['F'] = objs

def do_every_generations(algorithm):
    
    if algorithm.problem.config.log_stats: test.log_stats(algorithm)
    if algorithm.problem.config.save_model: test.save_model(algorithm)
