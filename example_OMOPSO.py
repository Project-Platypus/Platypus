from platypus.algorithms import OMOPSO
from platypus.problems import DTLZ2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

problem = DTLZ2(12, 3)
 
algorithm = OMOPSO(problem,
                   epsilons=[0.05],
                   mutation_probability = 1.0 / problem.nvars)
algorithm.run(10000)
 
for solution in algorithm.result:
    print solution.objectives
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result])
plt.show()
