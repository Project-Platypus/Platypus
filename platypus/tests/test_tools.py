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
import functools
from ..tools import *
from .test_core import createSolution


class TestBinary(unittest.TestCase):
    
    EXPECTED = {
        0 : { "binary" : (0, 0, 0, 0), "gray" : (0, 0, 0, 0)},
        1 : { "binary" : (0, 0, 0, 1), "gray" : (0, 0, 0, 1)},
        2 : { "binary" : (0, 0, 1, 0), "gray" : (0, 0, 1, 1)},
        3 : { "binary" : (0, 0, 1, 1), "gray" : (0, 0, 1, 0)},
        4 : { "binary" : (0, 1, 0, 0), "gray" : (0, 1, 1, 0)},
        5 : { "binary" : (0, 1, 0, 1), "gray" : (0, 1, 1, 1)},
        6 : { "binary" : (0, 1, 1, 0), "gray" : (0, 1, 0, 1)},
        7 : { "binary" : (0, 1, 1, 1), "gray" : (0, 1, 0, 0)},
        8 : { "binary" : (1, 0, 0, 0), "gray" : (1, 1, 0, 0)},
        9 : { "binary" : (1, 0, 0, 1), "gray" : (1, 1, 0, 1)},
        10 : { "binary" : (1, 0, 1, 0), "gray" : (1, 1, 1, 1)},
        11 : { "binary" : (1, 0, 1, 1), "gray" : (1, 1, 1, 0)},
        12 : { "binary" : (1, 1, 0, 0), "gray" : (1, 0, 1, 0)},
        13 : { "binary" : (1, 1, 0, 1), "gray" : (1, 0, 1, 1)},
        14 : { "binary" : (1, 1, 1, 0), "gray" : (1, 0, 0, 1)},
        15 : { "binary" : (1, 1, 1, 1), "gray" : (1, 0, 0, 0)},
    }
    
    def assertBinEqual(self, b1, b2):
        self.assertEqual(len(b1), len(b2))
        
        for i in range(len(b1)):
            self.assertEqual(bool(b1[i]), bool(b2[i]))
    
    def test_int2bin(self):
        self.assertBinEqual([], int2bin(0, 0))
        self.assertBinEqual([0], int2bin(0, 1))
        
        for i in range(16):
            self.assertBinEqual(self.EXPECTED[i]["binary"], int2bin(i, 4))
        
    def test_bin2int(self):
        self.assertEqual(0, bin2int([]))
        self.assertEqual(0, bin2int([0]))

        for i in range(16):
            self.assertEqual(i, bin2int(self.EXPECTED[i]["binary"]))
        
    def test_bin2gray(self):
        for i in range(16):
            self.assertBinEqual(self.EXPECTED[i]["gray"], bin2gray(int2bin(i, 4)))
        
    def test_gray2bin(self):
        for i in range(16):
            self.assertBinEqual(self.EXPECTED[i]["binary"], gray2bin(self.EXPECTED[i]["gray"]))
            

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

class TestDictMethods(unittest.TestCase):

    def test_remove_keys(self):
        self.assertEqual({}, remove_keys({}))
        self.assertEqual({}, remove_keys({}, "a", "b"))
        self.assertEqual({}, remove_keys({"a" : "remove"}, "a", "b"))
        self.assertEqual({"c" : "keep"}, remove_keys({"a" : "remove", "c" : "keep"}, "a", "b"))
        self.assertEqual({"a" : "keep"}, remove_keys({"a" : "keep"}))

    def test_only_keys(self):
        self.assertEqual({}, only_keys({}))
        self.assertEqual({}, only_keys({}, "a", "b"))
        self.assertEqual({"a" : "keep"}, only_keys({"a" : "keep", "b" : "remove"}, "a"))

    def _test_func_pos(self, a):
        pass

    def _test_func_def(self, a=5):
        pass

    def test_keys_for(self):
        self.assertEqual({}, only_keys_for({}, self._test_func_pos))
        self.assertEqual({"a" : "keep"}, only_keys_for({"a" : "keep", "b" : "remove"}, self._test_func_pos))
        self.assertEqual({}, only_keys_for({}, self._test_func_def))
        self.assertEqual({"a" : "keep"}, only_keys_for({"a" : "keep", "b" : "remove"}, self._test_func_def))