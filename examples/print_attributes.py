from platypus import NSGAII, DTLZ2, normalize

problem = DTLZ2()

algorithm = NSGAII(problem)
algorithm.run(10000)

# Normalize the results.
result = algorithm.result
min_bounds, max_bounds = normalize(result)

# Display the attributes for one of the solutions.
print(f"Variables: {result[0].variables}")
print(f"Objectives: {result[0].objectives}")
print(f"Constraints: {result[0].constraints}")
print(f"Constraint Violation: {result[0].constraint_violation}")
print(f"Is Evaluated? {result[0].evaluated}")
print(f"Is Feasible? {result[0].feasible}")
print(f"Normalized Objectives: {result[0].normalized_objectives}")
