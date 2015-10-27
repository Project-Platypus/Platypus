from platypus.algorithms import NSGAII
from platypus.core import Problem, evaluator, Archive, nondominated
from platypus.types import Real

class Belegundu(Problem):

    def __init__(self):
        super(Belegundu, self).__init__(2, 2, 2)
        self.types[:] = [Real(0, 5), Real(0, 3)]
        self.constraints[:] = "<=0"
    
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[0]
        y = solution.variables[1]
        solution.objectives[:] = [-2*x + y, 2*x + y]
        solution.constraints[:] = [-x + y - 1, x + y - 7]

algorithm = NSGAII(Belegundu())
algorithm.run(20000)

import matplotlib.pyplot as plt
plt.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])
plt.xlabel("$f_1(x)$")
plt.ylabel("$f_2(x)$")
plt.show()

algorithm.result = nondominated(algorithm.result)

import matplotlib.pyplot as plt
plt.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])
plt.xlabel("$f_1(x)$")
plt.ylabel("$f_2(x)$")
plt.show()

import matplotlib.pyplot as plt
plt.scatter([s.objectives[0] for s in algorithm.result if s.feasible],
            [s.objectives[1] for s in algorithm.result if s.feasible])
plt.xlabel("$f_1(x)$")
plt.ylabel("$f_2(x)$")
plt.show()