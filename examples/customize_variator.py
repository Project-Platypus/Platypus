from platypus import DTLZ2, NSGAII, PCX

problem = DTLZ2()

algorithm = NSGAII(problem, variator=PCX())
algorithm.run(10000)
