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
import pytest

from ..config import PlatypusConfig
from ..evaluator import MapEvaluator, SubmitEvaluator
from ..operators import PM, SBX, GAOperator
from ..problems import DTLZ2
from ..types import Real


@pytest.fixture
def problem():
    return DTLZ2()

def test_default_variator(problem):
    assert isinstance(PlatypusConfig.default_variator(problem), GAOperator)

def test_default_variator_reassigned(problem):
    originalVariator = PlatypusConfig.default_variator(Real)

    try:
        PlatypusConfig.register_default_variator(Real, SBX())
        assert isinstance(PlatypusConfig.default_variator(problem), SBX)
    finally:
        PlatypusConfig.register_default_variator(Real, originalVariator)

def test_default_mutator(problem):
    assert isinstance(PlatypusConfig.default_mutator(problem), PM)

def test_default_mutator_reassigned(problem):
    originalMutator = PlatypusConfig.default_mutator(Real)

    try:
        PlatypusConfig.register_default_mutator(Real, PM())
        assert isinstance(PlatypusConfig.default_mutator(problem), PM)
    finally:
        PlatypusConfig.register_default_mutator(Real, originalMutator)

def test_default_evaluator():
    assert isinstance(PlatypusConfig.default_evaluator, MapEvaluator)

def test_default_evaluator_reassigned():
    originalEvaluator = PlatypusConfig.default_evaluator

    try:
        PlatypusConfig.default_evaluator = SubmitEvaluator(lambda x: x)
        assert isinstance(PlatypusConfig.default_evaluator, SubmitEvaluator)
    finally:
        PlatypusConfig.default_evaluator = originalEvaluator

def test_version():
    assert PlatypusConfig.version is not None
