from platypus.algorithms import CMAES
from platypus.problems import DTLZ2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

problem = DTLZ2(3)

algorithm = CMAES(problem, diagonal_iterations=0, epsilons=[0.1])

algorithm.run(20000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result])
plt.show()
