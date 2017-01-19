from platypus.algorithms import NSGAII, NSGAIII
from platypus.problems import DTLZ2
from platypus.indicators import Hypervolume
from platypus.experimenter import experiment, calculate, display

if __name__ == "__main__":
    algorithms = [NSGAII, (NSGAIII, {"divisions_outer":12})]
    problems = [DTLZ2(3)]

    # run the experiment
    results = experiment(algorithms, problems, nfe=10000, seeds=10)

    # calculate the hypervolume indicator
    hyp = Hypervolume(minimum=[0, 0, 0], maximum=[1, 1, 1])
    hyp_result = calculate(results, hyp)
    display(hyp_result, ndigits=3)