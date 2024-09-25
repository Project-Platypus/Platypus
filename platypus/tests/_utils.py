import pytest

from ..algorithms import Algorithm
from ..core import Archive, FixedLengthArray, Problem, Solution


def createSolution(*args):
    problem = Problem(0, len(args))
    solution = Solution(problem)
    solution.objectives[:] = [float(x) for x in args]
    return solution

def similar(expected, actual, epsilon=0.0000001):
    if isinstance(expected, Algorithm) and isinstance(actual, Algorithm):
        assert type(expected) is type(actual)
        assert expected.nfe == actual.nfe
        similar(expected.result, actual.result)
    elif isinstance(expected, Solution) and isinstance(actual, Solution):
        similar(expected.variables, actual.variables, epsilon)
        similar(expected.objectives, actual.objectives, epsilon)
        similar(expected.constraints, actual.constraints, epsilon)
    elif isinstance(expected, (list, FixedLengthArray, Archive)) and isinstance(actual, (list, FixedLengthArray, Archive)):
        for (x, y) in zip(expected, actual):
            similar(x, y, epsilon)
    else:
        assert pytest.approx(expected, abs=epsilon) == actual

def assertBinEqual(b1, b2):
    assert len(b1) == len(b2)
    for i in range(len(b1)):
        assert bool(b1[i]) == bool(b2[i])
