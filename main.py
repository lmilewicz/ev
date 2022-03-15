# To disable Tensorflow warnings:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.soo.nonconvex.isres import ISRES

from pymoo.optimize import minimize

import evolution
from config import Config

config = Config()

def main():
    problem = evolution.EVProblem(config)

    if config.algorithm == 'NSGA2': # https://pymoo.org/algorithms/moo/nsga2.html
        algorithm = NSGA2(pop_size=config.pop_size,
                            sampling=evolution.SamplingAll(),
                            mutation=evolution.MutationAll(),
                            eliminate_duplicates=True)
                            
    elif config.algorithm == 'UNSGA3': # https://pymoo.org/algorithms/moo/unsga3.html
        ref_dirs = np.array([[0.0, 0.0]])
        algorithm = UNSGA3(ref_dirs,
                            sampling=evolution.SamplingAll(),
                            mutation=evolution.MutationAll(),
                            eliminate_duplicates=True)

    elif config.algorithm == 'ISRES': # https://pymoo.org/algorithms/soo/isres.html
        algorithm = ISRES(pop_size=config.pop_size, 
                            rule=1.0 / 7.0, 
                            gamma=0.85, 
                            alpha=0.2)
    else:
        raise ValueError("Algorithm not chosen!")

    res = minimize(problem, 
                    algorithm, 
                    callback=evolution.do_every_generations, 
                    termination=config.termination)


if __name__ == "__main__":
    main()