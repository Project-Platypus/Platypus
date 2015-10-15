from platypus.algorithms import NSGAII
from platypus.operators import TournamentSelector, PM, SBX, GAOperator
from platypus.problems import DTLZ2

problem = DTLZ2(12, 3)

algorithm = NSGAII(problem,
                   population_size = 100,
                   selector = TournamentSelector(2),
                   variator = GAOperator(SBX(1.0), PM(1.0 / problem.nvars)))

algorithm.run(10000)

for solution in algorithm.result:
    print solution.objectives
