from platypus import GeneticAlgorithm, Problem, Constraint, Binary, nondominated, unique

# This simple example has an optimal value of 15 when picking items 1 and 4.
items = 7
capacity = 9
weights = [2, 3, 6, 7, 5, 9, 4]
profits = [6, 5, 8, 9, 6, 7, 3]
    
def knapsack(x):
    selection = x[0]
    total_weight = sum([weights[i] if selection[i] else 0 for i in range(items)])
    total_profit = sum([profits[i] if selection[i] else 0 for i in range(items)])
    
    return total_profit, total_weight

problem = Problem(1, 1, 1)
problem.types[0] = Binary(items)
problem.directions[0] = Problem.MAXIMIZE
problem.constraints[0] = Constraint("<=", capacity)
problem.function = knapsack

algorithm = GeneticAlgorithm(problem)
algorithm.run(10000)

for solution in unique(nondominated(algorithm.result)):
    print(solution.variables, solution.objectives)