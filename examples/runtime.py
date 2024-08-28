import matplotlib.pyplot as plt
from platypus import NSGAII, DTLZ2, Hypervolume, load_objectives

problem = DTLZ2()
algorithm = NSGAII(problem)

# Collect the population at each generation.
runtime = {}
algorithm.run(10000, callback=lambda a: runtime.update({a.nfe: a.result}))

# Compute the hypervolume for each generation.
hypervolume = {}
ref_set = load_objectives("examples/DTLZ2.2D.pf", problem)
hyp = Hypervolume(reference_set=ref_set)

for nfe, result in runtime.items():
    hypervolume[nfe] = hyp.calculate(result)

# Plot the results using matplotlib
plt.plot(hypervolume.keys(), hypervolume.values())
plt.xlabel("NFE")
plt.ylabel("Hypervolume")
plt.show()
