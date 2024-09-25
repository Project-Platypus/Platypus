from platypus import NSGAII, DTLZ2, SaveResultsExtension

problem = DTLZ2()

algorithm = NSGAII(problem)
algorithm.add_extension(SaveResultsExtension("NSGAII_DTLZ2_{nfe}.json", frequency=1000))
algorithm.run(10000)
