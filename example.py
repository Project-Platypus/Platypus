#from platypus.algorithms import NSGAII
#from platypus.problems import DTLZ2
from platypus.core import Problem
from platypus.types import Real
from platypus.algorithms import NSGAII

def myfunction(variables):
    # ...
    return (objs, constrs)

problem = Problem(2)
problem.variables = [Real(0, 1)]*11
problem.function = myfunction

algorithm = NSGAII(problem)
algorithm.operator = GAOperator(SBX(1.0, 15.0),
                                PM(1.0 / 11.0, 20.0))
algorithm.run(10000)

for solution in algorithm.result:
    print solution


#algorithm = NSGAII(problem)
#algorithm.run(10000)

#result = algorithm.result