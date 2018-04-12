# Copyright 2015-2018 David Hadka
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
import unittest
from mock import patch
from ..core import Problem, Solution
from ..types import Permutation
from ..operators import Swap

class TestSwap(unittest.TestCase):
    
    def test_swap10(self):
        problem = Problem(1, 0)
        problem.types[0] = Permutation(range(10))
        
        solution = Solution(problem)
        solution.variables[0] = list(range(10))
        
        with patch('random.randrange', side_effect=[2, 4]):
            result = Swap(1.0).mutate(solution)
        
        self.assertEqual(result.variables[0][2], 4)
        self.assertEqual(result.variables[0][4], 2)
        self.assertEqual(solution.variables[0][2], 2)
        self.assertEqual(solution.variables[0][4], 4)
        
    def test_swap2a(self):
        problem = Problem(1, 0)
        problem.types[0] = Permutation(range(2))
        
        solution = Solution(problem)
        solution.variables[0] = list(range(2))
        
        with patch('random.randrange', side_effect=[0, 1]):
            result = Swap(1.0).mutate(solution)
        
        self.assertEqual(result.variables[0][0], 1)
        self.assertEqual(result.variables[0][1], 0)

    def test_swap2b(self):
        problem = Problem(1, 0)
        problem.types[0] = Permutation(range(2))
        
        solution = Solution(problem)
        solution.variables[0] = list(range(2))
        
        with patch('random.randrange', side_effect=[1, 1, 0]):
            result = Swap(1.0).mutate(solution)
        
        self.assertEqual(result.variables[0][0], 1)
        self.assertEqual(result.variables[0][1], 0)
        
    def test_swap1(self):
        problem = Problem(1, 0)
        problem.types[0] = Permutation(range(1))
        
        solution = Solution(problem)
        solution.variables[0] = list(range(1))
        
        with patch('random.randrange', side_effect=[0, 0]):
            result = Swap(1.0).mutate(solution)
        
        self.assertEqual(result.variables[0][0], 0)      