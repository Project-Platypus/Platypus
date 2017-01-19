from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real

def schaffer(x):
   return [x[0]**2, (x[0]-2)**2]

problem = Problem(1, 2)
problem.types[:] = Real(-10, 10)
problem.function = schaffer

algorithm = NSGAII(problem)
algorithm.run(10000)