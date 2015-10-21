from platypus.problems import WFG9
from platypus.core import Solution, Archive, EpsilonDominance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

problem = WFG9(3)
result = Archive(EpsilonDominance([0.05]))

for _ in range(10000):
    result.append(problem.random())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in result],
           [s.objectives[1] for s in result],
           [s.objectives[2] for s in result])
plt.show()
