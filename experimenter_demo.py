from platypus.algorithms import NSGAII, NSGAIII
from platypus.problems import DTLZ2
from platypus.indicators import Hypervolume
from platypus.experimenter import experiment, calculate, display
from multiprocessing import Pool, freeze_support

if __name__ == "__main__":
    freeze_support() # required on Windows
    pool = Pool(6)
    
    algorithms = [NSGAII, (NSGAIII, {"divisions_outer":12})]
    problems = [DTLZ2]
    
    results = experiment(algorithms, problems, nfe=10000, map=pool.map)

    hyp = Hypervolume(minimum=[0, 0], maximum=[1, 1])
    hyp_result = calculate(results, hyp, map=pool.map)
    display(hyp_result)
    
    pool.close()
    pool.join()