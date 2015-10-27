from platypus.algorithms import *
from platypus.problems import DTLZ2
from multiprocessing import Pool
import matplotlib.pyplot as plt

# setup the comparison
problem = DTLZ2()
algorithms = [NSGAII(problem),
              NSGAIII(problem, divisions_outer=24),
              CMAES(problem, epsilons=[0.01]),
              GDE3(problem),
              IBEA(problem),
              MOEAD(problem),
              OMOPSO(problem, epsilons=[0.01]),
              SMPSO(problem),
              SPEA2(problem),
              EpsilonMOEA(problem, epsilons=[0.01])]

# run the algorithms in parallel
#pool = Pool(processes=8)
map(operator.methodcaller("run", 10000), algorithms)
    
# generate the result plot
def to_points(solutions):
    return [s.objectives[0] for s in solutions], [s.objectives[1] for s in solutions]

fig, axarr = plt.subplots(2, 5)

for i in range(len(algorithms)):
    axarr[i/5, i%5].scatter(*to_points(algorithms[i].result))
    axarr[i/5, i%5].set_title(algorithms[i].__class__.__name__)


plt.show()
