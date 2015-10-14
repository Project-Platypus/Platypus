#from platypus.algorithms import NSGAII
#from platypus.problems import DTLZ2
from platypus.core import Problem, Solution, nondominated_sort
from platypus.types import Real
from platypus.algorithms import NSGAII, GDE3
from platypus.operators import TournamentSelector, RandomGenerator, PM, SBX, GAOperator
from platypus.problems import DTLZ2
import operator
#from platypus.algorithms import NSGAII

problem = DTLZ2()

# algorithm = NSGAII(problem,
#                    population_size = 100,
#                    generator = RandomGenerator(),
#                    selector = TournamentSelector(2),
#                    variator = GAOperator(SBX(1.0), PM(1.0 / problem.nvars)))

algorithm = GDE3(problem,
                 population_size = 100,
                 generator = RandomGenerator(),
                 variator = GAOperator(SBX(1.0), PM(1.0 / problem.nvars)))

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