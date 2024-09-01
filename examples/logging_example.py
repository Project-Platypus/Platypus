from platypus import NSGAII, DTLZ2
import logging

# set log level so all messages are displayed
logging.basicConfig(level=logging.DEBUG)

# select the problem
problem = DTLZ2()

# create the optimization algorithm
algorithm = NSGAII(problem)

# optimize the problem using 10,000 function evaluations
algorithm.run(10000)

# display the results
for solution in algorithm.result:
    print(solution.objectives)
