from platypus import *

if __name__ == "__main__":
    algorithms = [NSGAII, (NSGAIII, {"divisions_outer":12})]
    problems = [DTLZ2(3)]
    
    with ProcessPoolEvaluator(4) as evaluator:
        results = experiment(algorithms, problems, nfe=10000, evaluator=evaluator)

        hyp = Hypervolume(minimum=[0, 0, 0], maximum=[1, 1, 1])
        hyp_result = calculate(results, hyp, evaluator=evaluator)
        display(hyp_result, ndigits=3)