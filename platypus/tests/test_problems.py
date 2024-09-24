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
import inspect
import importlib
from ..core import Problem
from ..problems import WFG, ZDT
from ..operators import RandomGenerator

problem_module = importlib.import_module("platypus.problems")
problem_filter = lambda x: inspect.isclass(x) and issubclass(x, Problem) and x not in (Problem, WFG, ZDT)
problems = [v for _, v in inspect.getmembers(problem_module, problem_filter)]

@pytest.mark.parametrize("problem", problems)
def test_problem(problem):
    p = problem()
    s = RandomGenerator().generate(p)
    p.evaluate(s)
