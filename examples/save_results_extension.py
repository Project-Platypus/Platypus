from platypus import NSGAII, DTLZ2, SaveResultsExtension

problem = DTLZ2()

algorithm = NSGAII(problem)
algorithm.add_extension(SaveResultsExtension("{algorithm}_{problem}_{nfe}.json", frequency=1000))
algorithm.run(10000)
