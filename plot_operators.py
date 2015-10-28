# Generates a plot showing the distributions of six real-valued operators.
import matplotlib.pyplot as plt
from platypus.core import Problem, Solution
from platypus.types import Real
from platypus.operators import SBX, PM, DifferentialEvolution, UM, PCX, UNDX, SPX, GAOperator
from platypus.tools import add, subtract

problem = Problem(2, 0)
problem.types[:] = Real(-1, 1)

solution1 = Solution(problem)
solution1.variables[:] = [-0.25, -0.25]

solution2 = Solution(problem)
solution2.variables[:] = [0.25, -0.25]

solution3 = Solution(problem)
solution3.variables[:] = [0.0, 0.25]

def generate(variator, parents):
    result = []
    
    while len(result) < 10000:
        result.extend(variator.evolve(parents))
        
    return to_points(result)

def to_points(solutions):
    return [s.variables[0] for s in solutions], [s.variables[1] for s in solutions]

fig, axarr = plt.subplots(2, 3)
options = {"s":0.1, "alpha":0.5}
parent_options = {"s":25, "color":"b"}

axarr[0, 0].scatter(*generate(GAOperator(SBX(1.0, 20.0), PM(0.5, 250.0)), [solution1, solution3]), **options)
axarr[0, 0].scatter(*to_points([solution1, solution3]), **parent_options)
axarr[0, 0].set_title("SBX")

axarr[0, 1].scatter(*generate(GAOperator(DifferentialEvolution(1.0, 1.0), PM(0.5, 100.0)), [solution3, solution2, solution1, solution3]), **options)
axarr[0, 1].scatter(*to_points([solution3, solution2, solution1, solution3]), **parent_options)
axarr[0, 1].set_title("DE")

axarr[0, 2].scatter(*generate(UM(0.5), [solution1]), **options)
axarr[0, 2].scatter(*to_points([solution1]), **parent_options)
axarr[0, 2].set_title("UM")

axarr[1, 0].scatter(*generate(PCX(3, 2), [solution1, solution2, solution3]), **options)
axarr[1, 0].scatter(*to_points([solution1, solution2, solution3]), **parent_options)
axarr[1, 0].set_title("PCX")

axarr[1, 1].scatter(*generate(UNDX(3, 2), [solution1, solution2, solution3]), **options)
axarr[1, 1].scatter(*to_points([solution1, solution2, solution3]), **parent_options)
axarr[1, 1].set_title("UNDX")

axarr[1, 2].scatter(*generate(SPX(3, 2), [solution1, solution2, solution3]), **options)
axarr[1, 2].scatter(*to_points([solution1, solution2, solution3]), **parent_options)
axarr[1, 2].set_title("SPX")

# add arrow annotations to DE
axarr[0, 1].annotate("",
                     xy=solution3.variables, xycoords='data',
                     xytext=solution1.variables, textcoords='data',
                     arrowprops=dict(arrowstyle="fancy",
                                     connectionstyle="arc3",
                                     color="0.75"))
axarr[0, 1].annotate("",
                     xy=add(solution2.variables, subtract(solution3.variables, solution1.variables)), xycoords='data',
                     xytext=solution2.variables, textcoords='data',
                     arrowprops=dict(arrowstyle="fancy",
                                     connectionstyle="arc3",
                                     color="0.75"))


for i in range(1):
    for j in range(3):
        axarr[i, j].set_xticklabels([])
        axarr[i, j].set_yticklabels([])
        axarr[i, j].autoscale(tight=True)

plt.show()