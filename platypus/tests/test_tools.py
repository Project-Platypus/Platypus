# Copyright 2015-2016 David Hadka
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
import functools
from ..tools import *
from .test_core import createSolution

class TestVectorAlgebra(unittest.TestCase):
    
    def test_dot(self):
        self.assertEqual(0, dot([0, 0, 0], [1, 1, 1]))
        self.assertEqual(1, dot([1, 0, 1], [0, 1, 1]))
        self.assertEqual(2, dot([1, 0, 1], [1, 1, 1]))
        
    def test_subtract(self):
        self.assertEquals([1, 0, -1], subtract([1, 1, 1], [0, 1, 2]))
        
    def test_multiply(self):
        self.assertEquals([0, 0, 0], multiply(0, [1, 1, 1]))
        self.assertEquals([.5, .5, .5], multiply(.5, [1, 1, 1]))
        
    def test_magnitude(self):
        self.assertEqual(0, magnitude([0, 0, 0]))
        self.assertAlmostEqual(1, magnitude([0.577, 0.577, 0.577]), delta=0.001)
    
    def test_distance1(self):
        line = [1, 1, 1]
        point = [2, 2, 2]
        self.assertEqual(0.0, point_line_dist(point, line))
        
    def test_distance2(self):
        line = [1, 1, 1]
        point = [0, 0, 1]
        self.assertAlmostEqual(0.816, point_line_dist(point, line), places=3)
        
    def test_distance3(self):
        line = [1, 0, 0]
        point = [0, 0, 1]
        self.assertEqual(1.0, point_line_dist(point, line))
        
    def test_lsolve1(self):
        A = [[1, 0], [0, 1]]
        b = [1, 1]
        self.assertEquals([1, 1], lsolve(A, b))
        
    def test_lsolve2(self):
        A = [[0.7, 0.3, 0.0], [0.1, 0.1, 0.7], [0.2, 0.1, 0.9]]
        b = [0.2, 0.3, 0.5]
        x = lsolve(A, b)
        self.assertAlmostEqual(0.965, x[0], delta=0.001)
        self.assertAlmostEqual(-1.586, x[1], delta=0.001)
        self.assertAlmostEqual(0.517, x[2], delta=0.001)
        
    def test_lsolve_singular(self):
        A = [[0.5, 0.5], [0.5, 0.5]]
        b = [1, 1]
        self.assertRaises(SingularError, functools.partial(lsolve, A, b))

class TestCombinatorics(unittest.TestCase):
    
    def test_choose(self):
        self.assertEqual(1, choose(0, 0))
        self.assertEqual(1, choose(1, 0))
        self.assertEqual(1, choose(5, 0))
        self.assertEqual(1, choose(1, 1))
        self.assertEqual(5, choose(5, 1))
        self.assertEqual(5, choose(5, 4))
        self.assertEqual(1, choose(5, 5))
        
class TestDistances(unittest.TestCase):
    
    def test_euclidean(self):
        self.assertEqual(0.0, euclidean_dist([1, 1], [1, 1]))
        self.assertAlmostEqual(1.414, euclidean_dist([0, 0], [1, 1]), delta=0.001)
        self.assertAlmostEqual(1.414, euclidean_dist([1, 1], [0, 0]), delta=0.001)
        
class TestDistanceMatrix(unittest.TestCase):
    
    def test(self):
        solutions = [createSolution(0, 1), createSolution(0.5, 0.5), createSolution(0.75, 0.25), createSolution(1, 0)]
        matrix = DistanceMatrix(solutions)
        
        self.assertAlmostEqual(0.353, matrix[1, 2], delta=0.001)
        self.assertAlmostEqual(0.353, matrix[2, 1], delta=0.001)
        self.assertAlmostEqual(0.353, matrix.kth_distance(2, 0), delta=0.001)
        self.assertAlmostEqual(0.353, matrix.kth_distance(1, 0), delta=0.001)
        self.assertEqual(2, matrix.find_most_crowded())
        matrix.remove_point(2)
        self.assertEqual(1, matrix.find_most_crowded())