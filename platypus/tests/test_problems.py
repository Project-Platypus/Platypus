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
from ..problems import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7, \
    WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9, \
    UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10, UF11, UF12, UF13, \
    CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10, \
    ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6
from ..operators import RandomGenerator

@pytest.mark.parametrize("problem", [
    DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7,
    WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9,
    UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10, UF11, UF12, UF13,
    CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10,
    ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6])
def test_problem(problem):
    p = problem()
    s = RandomGenerator().generate(p)
    p.evaluate(s)
