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

import math
import tempfile
import unittest
from .test_core import createSolution
from ..algorithms import NSGAII
from ..core import FixedLengthArray
from ..problems import DTLZ2
from ..io import save_objectives, load_objectives, save_json, load_json

class TestObjectives(unittest.TestCase):

    def test(self):
        s1 = createSolution(0.0, 1.0)
        s2 = createSolution(1.0, 0.0)
        expected = [s1, s2]

        with tempfile.NamedTemporaryFile() as f:
            save_objectives(f.name, expected)
            actual = load_objectives(f.name, s1.problem)

        self.assertEqual(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertEqual(expected[i].objectives, actual[i].objectives)

class TestJSON(unittest.TestCase):

    def assertSimilar(self, a, b, epsilon=0.0000001):
        if isinstance(a, (list, FixedLengthArray)) and isinstance(b, (list, FixedLengthArray)):
            for (x, y) in zip(a, b):
                self.assertSimilar(x, y, epsilon)
        else:
            self.assertLessEqual(math.fabs(b - a), epsilon)

    def testSolutions(self):
        s1 = createSolution(0.0, 1.0)
        s2 = createSolution(1.0, 0.0)
        expected = [s1, s2]

        with tempfile.NamedTemporaryFile() as f:
            save_json(f.name, expected)
            actual = load_json(f.name)

        self.assertEqual(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertIsNotNone(actual[i].problem)
            self.assertSimilar(expected[i].variables, actual[i].variables)
            self.assertSimilar(expected[i].objectives, actual[i].objectives)
            self.assertSimilar(expected[i].constraints, actual[i].constraints)

    def testAlgorithm(self):
        problem = DTLZ2()
        algorithm = NSGAII(problem)
        algorithm.run(10000)

        expected = algorithm.result

        with tempfile.NamedTemporaryFile() as f:
            save_json(f.name, algorithm)
            actual = load_json(f.name)

        self.assertEqual(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertIsNotNone(actual[i].problem)
            self.assertSimilar(expected[i].variables, actual[i].variables)
            self.assertSimilar(expected[i].objectives, actual[i].objectives)
            self.assertSimilar(expected[i].constraints, actual[i].constraints)
