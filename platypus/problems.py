import math
import operator
from platypus.core import Problem
from platypus.types import Real

class DTLZ2(Problem):
    
    def __init__(self, nvars=11, nobjs=2):
        super(DTLZ2, self).__init__(nvars, nobjs)
        self.types[:] = Real(0, 1)
        self.directions[:] = Problem.MINIMIZE
        
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = sum(map(lambda x : pow(x - 0.5, 2.0), solution.variables[self.nvars-k:]))
        f = [1.0+g]*self.nobjs
        
        for i in range(self.nobjs):
            f[i] *= reduce(operator.mul,
                           [math.cos(0.5 * math.pi * x) for x in solution.variables[:self.nobjs-i-1]],
                           1)
            
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution.variables[self.nobjs-i-1])
        
        solution.objectives[:] = f
        solution.evaluated = True