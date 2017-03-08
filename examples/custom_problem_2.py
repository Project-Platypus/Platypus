from platypus import NSGAII, Problem, Real

class Schaffer(Problem):

    def __init__(self):
        super(Schaffer, self).__init__(1, 2)
        self.types[:] = Real(-10, 10)
    
    def evaluate(self, solution):
        x = solution.variables[:]
        solution.objectives[:] = [x[0]**2, (x[0]-2)**2]

algorithm = NSGAII(Schaffer())
algorithm.run(10000)
