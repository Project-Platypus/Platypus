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

import tempfile
import unittest
from ._utils import SolutionMixin
from ..algorithms import NSGAII
from ..problems import DTLZ2
from ..io import save_objectives, load_objectives, save_json, load_json, \
    save_state, load_state

class TestObjectives(SolutionMixin, unittest.TestCase):

    def test(self):
        s1 = self.createSolution(0.0, 1.0)
        s2 = self.createSolution(1.0, 0.0)
        expected = [s1, s2]

        with tempfile.NamedTemporaryFile() as f:
            save_objectives(f.name, expected)
            actual = load_objectives(f.name, s1.problem)

        self.assertEqual(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertEqual(expected[i].objectives, actual[i].objectives)

class TestJSON(SolutionMixin, unittest.TestCase):

    def test_solutions(self):
        s1 = self.createSolution(0.0, 1.0)
        s2 = self.createSolution(1.0, 0.0)
        expected = [s1, s2]

        with tempfile.NamedTemporaryFile() as f:
            save_json(f.name, expected)
            actual = load_json(f.name)

        self.assertEqual(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertIsNotNone(actual[i].problem)
            self.assertSimilar(expected[i], actual[i])

    def test_algorithm(self):
        problem = DTLZ2()
        algorithm = NSGAII(problem)
        algorithm.run(1000)

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

class TestState(SolutionMixin, unittest.TestCase):

    def run_test(self, json):
        problem = DTLZ2()
        original = NSGAII(problem)

        with tempfile.NamedTemporaryFile() as f:
            save_state(f.name, original, json=json)

            original.run(1000)

            copy = load_state(f.name)
            copy.run(1000)

            self.assertEqual(original.nfe, copy.nfe)

            expected = original.result
            actual = copy.result

            self.assertEqual(len(expected), len(actual))

            for i in range(len(expected)):
                self.assertSimilar(expected[i], actual[i])

    def test_binary(self):
        self.run_test(False)

    def test_json(self):
        self.run_test(True)
