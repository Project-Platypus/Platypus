from platypus import NSGAII, DTLZ2, PCX

problem = DTLZ2()

algorithm = NSGAII(problem, variator=PCX())
algorithm.run(10000)
