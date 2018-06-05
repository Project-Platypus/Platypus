from platypus import GeneticAlgorithm, Problem, Permutation, nondominated, unique

def ordering(x):
    # x[0] is the permutation, this calculates the difference between the permutation and an ordered list
    return sum([abs(p_i - i) for i, p_i in enumerate(x[0])])

problem = Problem(1, 1)
problem.types[0] = Permutation(range(10)) # Permutation of elements [0, 1, ..., 9]
problem.function = ordering

algorithm = GeneticAlgorithm(problem)
algorithm.run(10000)

for solution in unique(nondominated(algorithm.result)):
    print(solution.variables, solution.objectives)
    
# Output:
# [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] [0]
