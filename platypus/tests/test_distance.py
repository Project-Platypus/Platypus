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
import unittest
from ..distance import euclidean_dist, manhattan_dist, DistanceMatrix
from .test_core import createSolution


class TestDistances(unittest.TestCase):

    def test_euclidean(self):
        self.assertEqual(0.0, euclidean_dist([1, 1], [1, 1]))
        self.assertAlmostEqual(1.414, euclidean_dist([0, 0], [1, 1]), delta=0.001)
        self.assertAlmostEqual(1.414, euclidean_dist([1, 1], [0, 0]), delta=0.001)

    def test_manhattan(self):
        self.assertEqual(0.0, manhattan_dist([1, 1], [1, 1]))
        self.assertAlmostEqual(2.0, manhattan_dist([0, 0], [1, 1]), delta=0.001)
        self.assertAlmostEqual(2.0, manhattan_dist([1, 1], [0, 0]), delta=0.001)

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
