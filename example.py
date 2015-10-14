#from platypus.algorithms import NSGAII
#from platypus.problems import DTLZ2
from platypus.core import Problem, Solution, nondominated_sort
from platypus.types import Real
from platypus.algorithms import NSGAII
from platypus.operators import TournamentSelector, RandomGenerator, PM
import operator
#from platypus.algorithms import NSGAII

def createSolution(problem, x):
    solution = Solution(problem)
    solution.variables = x
    solution.evaluate()
    return solution

problem = Problem(1, 2, 0)
problem.types[:] = Real(0, 2)
problem.directions[:] = Problem.MINIMIZE
problem.function = lambda x : [x[0]**2, (x[0]-2)**2]

algorithm = NSGAII(problem,
                   population_size = 100,
                   generator = RandomGenerator(),
                   selector = TournamentSelector(2),
                   variator = PM(1.0))

algorithm.run(10000)

for solution in algorithm.result:
    print solution

#problem.function = #...

#algorithm = NSGAII(problem)
#algorithm.run(10000)

#for solution in algorithm.result:
#    print solution


#algorithm = NSGAII(problem)
#algorithm.run(10000)

#result = algorithm.result