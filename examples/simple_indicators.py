import math
from platypus import *

# Define the problem definition.
problem = DTLZ2()

# Instantiate the optimization algorithm.
algorithm = NSGAII(problem)

# Optimize the problem using 10,000 function evaluations.
algorithm.run(10000)

# Create the reference set.  For 2-objective DTLZ2, the reference set
# solutions must satisfy the equation x^2 + y^2 = 1.
ref_set = []

for i in range(100):
    solution = Solution(problem)
    solution.objectives[:] = [i / 100.0, math.sqrt(1.0 - (i / 100.0)**2)]
    ref_set.append(solution)

# Calculate the performance metrics.
hyp = Hypervolume(reference_set = ref_set)
print("Hypervolume:", hyp.calculate(algorithm.result))

gd = GenerationalDistance(reference_set = ref_set)
print("GD:", gd.calculate(algorithm.result))

igd = InvertedGenerationalDistance(reference_set = ref_set)
print("IGD:", igd.calculate(algorithm.result))

aei = EpsilonIndicator(reference_set = ref_set)
print("Eps-Indicator:", aei.calculate(algorithm.result))

spacing = Spacing()
print("Spacing:", spacing.calculate(algorithm.result))
