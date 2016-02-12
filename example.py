from platypus import NSGAIII, DTLZ2

# define the problem definition
problem = DTLZ2(3)

# instantiate the optimization algorithm
algorithm = NSGAIII(problem, divisions_outer=12)

# optimize the problem using 10,000 function evaluations
algorithm.run(10000)

# plot the results using matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result])
plt.show()