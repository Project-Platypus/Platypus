import matplotlib.pyplot as plt
from platypus import NSGAII, DTLZ2, Hypervolume, load_objectives

problem = DTLZ2()
algorithm = NSGAII(problem)

# Collect the population at each generation.
results = {}
algorithm.run(10000, callback=lambda a: results.update({a.nfe: {"population": a.result}}))

# Compute the hypervolume at each generation.
ref_set = load_objectives("examples/DTLZ2.2D.pf", problem)
hyp = Hypervolume(reference_set=ref_set)

for nfe, data in results.items():
    data["hypervolume"] = hyp.calculate(data["population"])

# Plot the results using matplotlib.
plt.plot(results.keys(),
         [x["hypervolume"] for x in results.values()])
plt.xlabel("NFE")
plt.ylabel("Hypervolume")
plt.show()
