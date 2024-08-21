from platypus import NSGAII, DTLZ2, PoolEvaluator
from platypus.mpipool import MPIPool
import sys
import logging

logging.basicConfig(level=logging.INFO)

# simulate an computationally expensive problem
class DTLZ2_Slow(DTLZ2):

    def evaluate(self, solution):
        sum(range(100000))
        super().evaluate(solution)

if __name__ == "__main__":
    problem = DTLZ2_Slow()

    with MPIPool() as pool:
        # only run the algorithm on the master process
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        # supply an evaluator to run in parallel
        with PoolEvaluator(pool) as evaluator:
            algorithm = NSGAII(problem, evaluator=evaluator)
            algorithm.run(10000)

        # display the results
        for solution in algorithm.result:
            print(solution.objectives)
