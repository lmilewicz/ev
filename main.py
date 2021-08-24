import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize

import evolution

def main():
    layers = 5
    max_connections = int((layers-1)*layers/2) # E.g. genome for 4 layers  - all connected: [1], [1, 1], [1, 1, 1]

    xl = np.zeros(max_connections)
    xu = np.ones(max_connections)

    problem = evolution.EVProblem(n_var=max_connections, n_obj=1, n_constr=0, xl=xl, xu=xu, layers=layers)

    _pop_size = 10
    algorithm = NSGA2(pop_size=_pop_size, 
                        n_offsprings=_pop_size, 
                        sampling=evolution.MySampling(),
                        mutation=evolution.MyMutation(),
                        eliminate_duplicates=True)

    res = minimize(problem, algorithm, callback=evolution.do_every_generations, termination=('n_gen', 10))


if __name__ == "__main__":
    main()