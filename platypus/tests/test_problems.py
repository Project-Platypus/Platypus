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
import importlib
import inspect
import pytest

from ..core import Problem
from ..operators import RandomGenerator
from ..problems import WFG, ZDT


def problem_filter(x):
    return inspect.isclass(x) and issubclass(x, Problem) and x not in (Problem, WFG, ZDT)

problem_module = importlib.import_module("platypus.problems")
problems = [v for _, v in inspect.getmembers(problem_module, problem_filter)]

@pytest.mark.parametrize("problem", problems)
def test_problem(problem):
    p = problem()
    s = RandomGenerator().generate(p)
    p.evaluate(s)
    assert all([x is not None for x in s.variables])
    assert all([x is not None for x in s.objectives])
    assert all([x is not None for x in s.constraints])
