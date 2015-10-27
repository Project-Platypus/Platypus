from platypus.algorithms import *
from platypus.problems import WFG2
from platypus.operators import GAOperator, SBX, PM
from multiprocessing import Pool
import matplotlib.pyplot as plt

# setup the comparison
problem = WFG2()
algorithms = [NSGAII(problem, variator=GAOperator(SBX(), PM())),
              NSGAIII(problem, divisions_outer=24, variator=GAOperator(SBX(), PM())),
              CMAES(problem, epsilons=[0.01]),
              GDE3(problem),
              IBEA(problem, variator=GAOperator(SBX(), PM())),
              MOEAD(problem, variator=GAOperator(SBX(), PM())),
              OMOPSO(problem, epsilons=[0.01]),
              SMPSO(problem, mutate=PM()),
              SPEA2(problem, variator=GAOperator(SBX(), PM())),
              EpsilonMOEA(problem, epsilons=[0.01], variator=GAOperator(SBX(), PM()))]

# run the algorithms in parallel
#pool = Pool(processes=8)
map(operator.methodcaller("run", 10000), algorithms)
    
# generate the result plot
def to_points(solutions):
    return [s.objectives[0] for s in solutions], [s.objectives[1] for s in solutions]

fig, axarr = plt.subplots(2, 5, sharex=True, sharey=True)

for i in range(len(algorithms)):
    ax = axarr[i/5, i%5]
    ax.scatter(*to_points(algorithms[i].result))
    ax.set_title(algorithms[i].__class__.__name__)
    #ax.set_xlim([0, 1])
    #ax.set_ylim([0, 1])

fig.text(0.5, 0.04, '$f_1(x)$', ha='center', va='center')
fig.text(0.04, 0.5, '$f_2(x)$', ha='center', va='center', rotation='vertical')

plt.locator_params(nbins=4)
plt.show()
