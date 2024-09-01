from platypus import NSGAII, DTLZ2
import matplotlib.pyplot as plt

# Select the problem.
problem = DTLZ2()

# Create the optimization algorithm.
algorithm = NSGAII(problem)

# Optimize the problem using 10,000 function evaluations
algorithm.run(10000)

# Plot the results using matplotlib
plt.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.xlabel("$f_1(x)$")
plt.ylabel("$f_2(x)$")
plt.show()
