# Copyright 2015-2024 David Hadka
#
# This file is part of Platypus, a Python module for designing and using
# evolutionary algorithms (EAs) and multiobjective evolutionary algorithms
# (MOEAs).
#
# Platypus is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Platypus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Platypus.  If not, see <http://www.gnu.org/licenses/>.
from unittest.mock import patch

from ..core import Problem, Solution
from ..operators import Swap
from ..types import Permutation


def test_swap_range10():
    problem = Problem(1, 0)
    problem.types[0] = Permutation(range(10))

    solution = Solution(problem)
    solution.variables[0] = list(range(10))

    with patch('random.randrange', side_effect=[2, 4]):
        result = Swap(1.0).mutate(solution)

    assert 4 == result.variables[0][2]
    assert 2 == result.variables[0][4]
    assert 2 == solution.variables[0][2]
    assert 4 == solution.variables[0][4]

def test_swap_range2_test1():
    problem = Problem(1, 0)
    problem.types[0] = Permutation(range(2))

    solution = Solution(problem)
    solution.variables[0] = list(range(2))

    with patch('random.randrange', side_effect=[0, 1]):
        result = Swap(1.0).mutate(solution)

    assert 1 == result.variables[0][0]
    assert 0 == result.variables[0][1]

def test_swap_range2_test2():
    problem = Problem(1, 0)
    problem.types[0] = Permutation(range(2))

    solution = Solution(problem)
    solution.variables[0] = list(range(2))

    with patch('random.randrange', side_effect=[1, 1, 0]):
        result = Swap(1.0).mutate(solution)

    assert 1 == result.variables[0][0]
    assert 0 == result.variables[0][1]

def test_swap_range1():
    problem = Problem(1, 0)
    problem.types[0] = Permutation(range(1))

    solution = Solution(problem)
    solution.variables[0] = list(range(1))

    with patch('random.randrange', side_effect=[0, 0]):
        result = Swap(1.0).mutate(solution)

    assert 0 == result.variables[0][0]
