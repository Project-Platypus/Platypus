from platypus import NSGAII, DTLZ2, Hypervolume, GenerationalDistance, \
    InvertedGenerationalDistance, EpsilonIndicator, Spacing, load_objectives

# Select the problem.
problem = DTLZ2()

# Create the optimization algorithm.
algorithm = NSGAII(problem)

# Optimize the problem using 10,000 function evaluations.
algorithm.run(10000)

# Load the reference set.
ref_set = load_objectives("examples/DTLZ2.2D.pf", problem)

# Calculate the performance metrics.
hyp = Hypervolume(reference_set=ref_set)
print("Hypervolume:", hyp.calculate(algorithm.result))

gd = GenerationalDistance(reference_set=ref_set)
print("GD:", gd.calculate(algorithm.result))

igd = InvertedGenerationalDistance(reference_set=ref_set)
print("IGD:", igd.calculate(algorithm.result))

aei = EpsilonIndicator(reference_set=ref_set)
print("Eps-Indicator:", aei.calculate(algorithm.result))

spacing = Spacing()
print("Spacing:", spacing.calculate(algorithm.result))
