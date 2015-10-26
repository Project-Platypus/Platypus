from platypus.algorithms import SMPSO
from platypus.operators import PM
from platypus.problems import DTLZ2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

problem = DTLZ2(3)
 
algorithm = SMPSO(problem,
                  mutate = PM(1.0 / problem.nvars, 20.0))
algorithm.run(10000)
 
for solution in algorithm.result:
    print solution.objectives
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result])
plt.show()
