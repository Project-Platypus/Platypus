from platypus import OMOPSO, DTLZ2

# define the problem
problem = DTLZ2()

# OMOPSO uses a non-uniform mutation operator that scales down the magnitude of the mutations
# as the run progresses.  This requires passing in max_iterations.
swarm_size = 100
max_iterations = 500

algorithm = OMOPSO(problem, epsilons=[0.01], swarm_size=swarm_size, max_iterations=max_iterations)
algorithm.run(swarm_size * max_iterations)

# display the results
for solution in algorithm.result:
    print(solution.objectives)
