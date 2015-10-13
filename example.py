#from platypus.algorithms import NSGAII
#from platypus.problems import DTLZ2
from platypus.core import Problem
from platypus.types import Real
from platypus.algorithms import NSGAII

problem = Problem(11, 2, 1)
problem.variables[:] = Real(0, 1)
problem.objectives[:] = Problem.MINIMIZE
problem.constraints[:] = ">=0"
problem.function = #...

algorithm = NSGAII(problem)
algorithm.run(10000)

for solution in algorithm.result:
    print solution


#algorithm = NSGAII(problem)
#algorithm.run(10000)

#result = algorithm.result