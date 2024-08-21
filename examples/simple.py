from platypus import NSGAII, DTLZ2

# Select the problem
problem = DTLZ2()

# Create the optimization algorithm.
algorithm = NSGAII(problem)

# Optimize the problem using 10,000 function evaluations.
algorithm.run(10000)

# Display the results.
for solution in algorithm.result:
    print(solution.objectives)
