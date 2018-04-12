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
import math
import unittest
from .test_core import createSolution
from ..indicators import GenerationalDistance, InvertedGenerationalDistance, \
    EpsilonIndicator, Spacing, Hypervolume
from ..core import Solution, Problem, POSITIVE_INFINITY

class TestGenerationalDistance(unittest.TestCase):
    
    def test(self):
        reference_set = [createSolution(0, 1), createSolution(1, 0)]
        gd = GenerationalDistance(reference_set)
        
        set = []
        self.assertEqual(POSITIVE_INFINITY, gd(set))
        
        set = [createSolution(0.0, 1.0)]
        self.assertEqual(0.0, gd(set))
        
        set = [createSolution(0.0, 1.0), createSolution(1.0, 0.0)]
        self.assertEqual(0.0, gd(set))
        
        set = [createSolution(2.0, 2.0)]
        self.assertEqual(math.sqrt(5.0), gd(set))
        
        set = [createSolution(0.5, 0.0), createSolution(0.0, 0.5)]
        self.assertEqual(math.sqrt(0.5)/2.0, gd(set))

class TestInvertedGenerationalDistance(unittest.TestCase):
    
    def test(self):
        reference_set = [createSolution(0, 1), createSolution(1, 0)]
        igd = InvertedGenerationalDistance(reference_set)
        
        set = []
        self.assertEqual(POSITIVE_INFINITY, igd(set))
        
        set = [createSolution(0.0, 1.0)]
        self.assertEqual(math.sqrt(2.0)/2.0, igd(set))
        
        set = [createSolution(0.0, 1.0), createSolution(1.0, 0.0)]
        self.assertEqual(0.0, igd(set))
        
        set = [createSolution(2.0, 2.0)]
        self.assertEqual(2.0*math.sqrt(5.0)/2.0, igd(set))

class TestEpsilonIndicator(unittest.TestCase):
    
    def test(self):
        reference_set = [createSolution(0, 1), createSolution(1, 0)]
        ei = EpsilonIndicator(reference_set)
        
        set = []
        self.assertEqual(POSITIVE_INFINITY, ei(set))
        
        set = [createSolution(0.0, 1.0)]
        self.assertEqual(1.0, ei(set))
        
        set = [createSolution(0.0, 1.0), createSolution(1.0, 0.0)]
        self.assertEqual(0.0, ei(set))
        
        set = [createSolution(2.0, 2.0)]
        self.assertEqual(2.0, ei(set))

class TestSpacing(unittest.TestCase):
    
    def test(self):
        sp = Spacing()
        
        set = []
        self.assertEqual(0.0, sp(set))
        
        set = [createSolution(0.5, 0.5)]
        self.assertEqual(0.0, sp(set))
        
        set = [createSolution(0.0, 1.0), createSolution(1.0, 0.0)]
        self.assertEqual(0.0, sp(set))
        
        set = [createSolution(0.0, 1.0), createSolution(0.5, 0.5), createSolution(1.0, 0.0)]
        self.assertEqual(0.0, sp(set))
        
        set = [createSolution(0.0, 1.0), createSolution(0.25, 0.75), createSolution(1.0, 0.0)]
        self.assertGreater(sp(set), 0.0)
        
class TestHypervolume(unittest.TestCase):
    
    def test(self):
        reference_set = [createSolution(0.0, 1.0), createSolution(1.0, 0.0)]
        hyp = Hypervolume(reference_set)
        
        set = []
        self.assertEqual(0.0, hyp(set))
        
        set = [createSolution(0.5, 0.5)]
        self.assertEqual(0.25, hyp(set))
        
        set = [createSolution(0.0, 0.0)]
        self.assertEqual(1.0, hyp(set))
        
        set = [createSolution(1.0, 1.0)]
        self.assertEqual(0.0, hyp(set))
        
        set = [createSolution(0.5, 0.0), createSolution(0.0, 0.5)]
        self.assertEqual(0.75, hyp(set))
        
    def test_maximize(self):
        reference_set = [createSolution(0.0, 1.0), createSolution(1.0, 0.0)]
        hyp = Hypervolume(reference_set)
        
        problem = Problem(0, 2)
        problem.directions[:] = Problem.MAXIMIZE
        s1 = Solution(problem)
        s2 = Solution(problem)
        s1.objectives[:] = [0.5, 1.0]
        s2.objectives[:] = [1.0, 0.5]
        
        self.assertEqual(0.75, hyp([s1, s2]))
        