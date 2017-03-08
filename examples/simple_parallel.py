from platypus import NSGAII, DTLZ2, ProcessPoolEvaluator

# simulate an computationally expensive problem
class DTLZ2_Slow(DTLZ2):
    
    def evaluate(self, solution):
        sum(range(1000000))
        super(DTLZ2_Slow, self).evaluate(solution)

if __name__ == "__main__":
    # define the problem definition
    problem = DTLZ2_Slow()
    
    # instantiate the optimization algorithm to run in parallel
    with ProcessPoolEvaluator(4) as evaluator:
        algorithm = NSGAII(problem, evaluator=evaluator)
        algorithm.run(10000)
    
    # display the results
    for solution in algorithm.result:
        print(solution.objectives)