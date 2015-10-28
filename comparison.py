from platypus.algorithms import *
from platypus.problems import DTLZ2
from platypus.indicators import hypervolume
from multiprocessing import Pool, freeze_support
import matplotlib.pyplot as plt
import pickle

def run(x):
    print "Started", x.__class__.__name__
    x.run(10000)
    print "Finished", x.__class__.__name__

if __name__ == '__main__':
    freeze_support()
    
    # setup the comparison
    problem = DTLZ2()
    pool = Pool(7)
    algorithms = [NSGAII(problem),
              NSGAIII(problem, divisions_outer=24),
              CMAES(problem, epsilons=[0.01]),
              GDE3(problem),
              IBEA(problem),
              MOEAD(problem),
              OMOPSO(problem, epsilons=[0.01]),
              SMPSO(problem),
              SPEA2(problem),
              EpsMOEA(problem, epsilons=[0.01])]
        
    # run the algorithms
    pool.map(run, algorithms)
    
    # generate the result plot
    def to_points(solutions):
        return [s.objectives[0] for s in solutions], [s.objectives[1] for s in solutions]

    hyp = hypervolume(minimum=[0,0], maximum=[1,1])
    fig, axarr = plt.subplots(2, 5)

    for i in range(len(algorithms)):
        axarr[i/5, i%5].scatter(*to_points(algorithms[i].result))
        axarr[i/5, i%5].set_title(algorithms[i].__class__.__name__)
        axarr[i/5, i%5].annotate("Hyp: " + str(round(hyp(algorithms[i].result), 3)),
                             xy=(0.9, 0.9),
                             xycoords='axes fraction',
                             horizontalalignment='right',
                             verticalalignment='bottom')

        plt.show()
