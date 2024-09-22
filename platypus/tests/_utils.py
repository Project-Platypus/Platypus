import math
from ..core import Problem, Solution, FixedLengthArray

class SolutionMixin:

    def createSolution(self, *args):
        problem = Problem(0, len(args))
        solution = Solution(problem)
        solution.objectives[:] = [float(x) for x in args]
        return solution

    def assertSimilar(self, a, b, epsilon=0.0000001):
        if isinstance(a, Solution) and isinstance(b, Solution):
            self.assertSimilar(a.variables, b.variables)
            self.assertSimilar(a.objectives, b.objectives)
            self.assertSimilar(a.constraints, b.constraints)
        elif isinstance(a, (list, FixedLengthArray)) and isinstance(b, (list, FixedLengthArray)):
            for (x, y) in zip(a, b):
                self.assertSimilar(x, y, epsilon)
        else:
            self.assertLessEqual(math.fabs(b - a), epsilon)
