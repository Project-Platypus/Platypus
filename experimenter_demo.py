from platypus.algorithms import NSGAII, NSGAIII
from platypus.problems import DTLZ2
from platypus.indicators import Hypervolume
from platypus.experimenter import experiment, calculate, display
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    algorithms = [NSGAII, (NSGAIII, {"divisions_outer":12})]
    problems = [DTLZ2]
    
    with ProcessPoolExecutor(6) as pool:
        results = experiment(algorithms, problems, nfe=10000, submit=pool.submit)

        hyp = Hypervolume(minimum=[0, 0], maximum=[1, 1])
        hyp_result = calculate(results, hyp, submit=pool.submit)
        display(hyp_result)