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
from ..config import PlatypusConfig
from ..evaluator import MapEvaluator, SubmitEvaluator
from ..operators import GAOperator, PM, SBX
from ..problems import DTLZ2
from ..types import Real

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.problem = DTLZ2(2)

    def test_default_variator(self):
        self.assertIsNotNone(PlatypusConfig.default_variator(self.problem))
        self.assertIsInstance(PlatypusConfig.default_variator(self.problem), GAOperator)

    def test_default_variator_reassigned(self):
        originalVariator = PlatypusConfig.default_variator(Real)

        PlatypusConfig.register_default_variator(Real, SBX())
        self.assertIsInstance(PlatypusConfig.default_variator(self.problem), SBX)

        PlatypusConfig.register_default_variator(Real, originalVariator)

    def test_default_mutator(self):
        self.assertIsNotNone(PlatypusConfig.default_mutator(self.problem))
        self.assertIsInstance(PlatypusConfig.default_mutator(self.problem), PM)

    def test_default_mutator_reassigned(self):
        originalMutator = PlatypusConfig.default_mutator(Real)

        PlatypusConfig.register_default_mutator(Real, PM())
        self.assertIsInstance(PlatypusConfig.default_mutator(self.problem), PM)

        PlatypusConfig.register_default_mutator(Real, originalMutator)

    def test_default_evaluator(self):
        self.assertIsNotNone(PlatypusConfig.default_evaluator)
        self.assertIsInstance(PlatypusConfig.default_evaluator, MapEvaluator)

    def test_default_evaluator_reassigned(self):
        originalEvaluator = PlatypusConfig.default_evaluator

        PlatypusConfig.default_evaluator = SubmitEvaluator(lambda x: x)
        self.assertIsInstance(PlatypusConfig.default_evaluator, SubmitEvaluator)

        PlatypusConfig.default_evaluator = originalEvaluator
