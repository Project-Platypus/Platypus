from platypus.algorithms import GDE3
from platypus.problems import DTLZ2

problem = DTLZ2()

algorithm = GDE3(problem, population_size = 100)
algorithm.run(10000)

for solution in algorithm.result:
    print solution
