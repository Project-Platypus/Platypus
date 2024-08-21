from platypus import NSGAIII, DTLZ2
import matplotlib.pyplot as plt

# Select the problem.
problem = DTLZ2(3)

# Create the optimization algorithm.
algorithm = NSGAIII(problem, divisions_outer=12)

# Optimize the problem using 10,000 function evaluations.
algorithm.run(10000)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result])
plt.show()
