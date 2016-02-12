from platypus import *
from multiprocessing import  freeze_support

if __name__ == "__main__":
    freeze_support() # required on Windows
    
    with MultiprocessingEvaluator() as evaluator:
        algorithms = [NSGAII, (NSGAIII, {"divisions_outer":12})]
        problems = [DTLZ2(3)]
        
        results = experiment(algorithms, problems, nfe=10000, evaluator=evaluator)
    
        hyp = Hypervolume(minimum=[0, 0, 0], maximum=[1, 1, 1])
        hyp_result = calculate(results, hyp, evaluator=evaluator)
        display(hyp_result, ndigits=3)