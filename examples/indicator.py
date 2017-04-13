import random
from platypus import (NSGAII, DTLZ2, Solution, EpsilonBoxArchive, GenerationalDistance, InvertedGenerationalDistance,
                      Hypervolume, EpsilonIndicator, Spacing)

# create the problem
problem = DTLZ2(3)

# solve it using NSGA-II
algorithm = NSGAII(problem)
algorithm.run(10000)

# generate the reference set for 3D DTLZ2
reference_set = EpsilonBoxArchive([0.02, 0.02, 0.02])

for _ in range(1000):
    solution = Solution(problem)
    solution.variables = [random.uniform(0,1) if i < problem.nobjs-1 else 0.5 for i in range(problem.nvars)]
    solution.evaluate()
    reference_set.add(solution)

# compute the indicators
gd = GenerationalDistance(reference_set)
print("Generational Distance:", gd.calculate(algorithm.result))

igd = InvertedGenerationalDistance(reference_set)
print("Inverted Generational Distance:", igd.calculate(algorithm.result))

hyp = Hypervolume(reference_set)
print("Hypervolume:", hyp.calculate(algorithm.result))

ei = EpsilonIndicator(reference_set)
print("Epsilon Indicator:", ei.calculate(algorithm.result))

sp = Spacing()
print("Spacing:", sp.calculate(algorithm.result))

# plot the result versus the reference set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter([s.objectives[0] for s in reference_set],
           [s.objectives[1] for s in reference_set],
           [s.objectives[2] for s in reference_set],
           c="red",
           edgecolors="none",
           label="Reference Set")
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result],
           c="blue",
           edgecolors="none",
           label = "NSGA-II Result")
ax.set_title("Reference Set")
ax.set_xlim([0, 1.1])
ax.set_ylim([0, 1.1])
ax.set_zlim([0, 1.1])
ax.set_xlabel("$f_1(x)$")
ax.set_ylabel("$f_2(x)$")
ax.set_zlabel("$f_3(x)$")
ax.view_init(elev=30.0, azim=15.0)
ax.locator_params(nbins=4)
ax.legend()
plt.show()
