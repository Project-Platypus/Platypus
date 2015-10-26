from platypus.algorithms import NSGAIII
from platypus.operators import TournamentSelector, PM, SBX, GAOperator
from platypus.problems import WFG1
from platypus.core import Archive, EpsilonDominance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

problem = WFG1(3)

# algorithm = NSGAIII(problem, 12,
#                    selector = TournamentSelector(2),
#                    variator = GAOperator(SBX(1.0), PM(1.0 / problem.nvars)))
# 
# algorithm.run(10000)

result = Archive(EpsilonDominance([0.1]))

for _ in range(10000):
    result += problem.random()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in result],
           [s.objectives[1] for s in result],
           [s.objectives[2] for s in result])
plt.show()
