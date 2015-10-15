from platypus.algorithms import SPEA2
from platypus.operators import PM, SBX, GAOperator
from platypus.problems import DTLZ2
import matplotlib.pyplot as plt

problem = DTLZ2()

algorithm = SPEA2(problem,
                  population_size = 100,
                  variator = GAOperator(SBX(1.0), PM(1.0 / problem.nvars)))

algorithm.run(10000)

# plot the results
plt.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])
plt.show()