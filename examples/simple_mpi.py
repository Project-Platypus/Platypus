from platypus.algorithms import NSGAII
from platypus.problems import DTLZ2
from platypus.evaluator import PoolEvaluator
from platypus.mpipool import MPIPool
import sys
import logging

logging.basicConfig(level=logging.INFO)

# simulate an computationally expensive problem
class DTLZ2_Slow(DTLZ2):
    
    def evaluate(self, solution):
        sum(range(1000000))
        super(DTLZ2_Slow, self).evaluate(solution)

if __name__ == "__main__":
    # define the problem definition
    problem = DTLZ2_Slow()
    pool = MPIPool()

    # only run the algorithm on the master process
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # instantiate the optimization algorithm to run in parallel
    with PoolEvaluator(pool) as evaluator:
        algorithm = NSGAII(problem, evaluator=evaluator)
        algorithm.run(10000)
    
    # display the results
    for solution in algorithm.result:
        print(solution.objectives)

    pool.close()
