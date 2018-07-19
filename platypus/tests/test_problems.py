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
import pickle
import unittest
import functools
from ..problems import *
from ..operators import RandomGenerator

class TestProblemsSimple(unittest.TestCase):

    def test_DTLZ1(self):
        self._run_test(DTLZ1(2))
    
    def test_DTLZ2(self):
        self._run_test(DTLZ2(2))
        
    def test_DTLZ3(self):
        self._run_test(DTLZ3(2))
        
    def test_DTLZ4(self):
        self._run_test(DTLZ4(2))
        
    def test_DTLZ7(self):
        self._run_test(DTLZ7(2))

    def test_WFG1(self):
        self._run_test(WFG1(2))

    def test_WFG2(self):
        self._run_test(WFG2(2))
        
    def test_WFG3(self):
        self._run_test(WFG3(2))
        
    def test_WFG4(self):
        self._run_test(WFG4(2))
        
    def test_WFG5(self):
        self._run_test(WFG5(2))
        
    def test_WFG6(self):
        self._run_test(WFG6(2))
        
    def test_WFG7(self):
        self._run_test(WFG7(2))
        
    def test_WFG8(self):
        self._run_test(WFG8(2))
        
    def test_WFG9(self):
        self._run_test(WFG9(2))

    def test_UF1(self):
        self._run_test(UF1())
        
    def test_UF2(self):
        self._run_test(UF2())
    
    def test_UF3(self):
        self._run_test(UF3())
        
    def test_UF4(self):
        self._run_test(UF4())
        
    def test_UF5(self):
        self._run_test(UF5())
        
    def test_UF6(self):
        self._run_test(UF6())
        
    def test_UF7(self):
        self._run_test(UF7())
        
    def test_UF8(self):
        self._run_test(UF8())
        
    def test_UF9(self):
        self._run_test(UF9())
        
    def test_UF10(self):
        self._run_test(UF10())
        
    def test_UF11(self):
        self._run_test(UF11())
        
    def test_UF12(self):
        self._run_test(UF12())
        
    def test_UF13(self):
        self._run_test(UF13())
        
    def test_CF1(self):
        self._run_test(CF1())
        
    def test_CF2(self):
        self._run_test(CF2())
        
    def test_CF3(self):
        self._run_test(CF3())
        
    def test_CF4(self):
        self._run_test(CF4())
        
    def test_CF5(self):
        self._run_test(CF5())
        
    def test_CF6(self):
        self._run_test(CF6())
        
    def test_CF7(self):
        self._run_test(CF7())
        
    def test_CF8(self):
        self._run_test(CF8())
        
    def test_CF9(self):
        self._run_test(CF9())
        
    def test_CF10(self):
        self._run_test(CF10())
        
    def test_ZDT1(self):
        self._run_test(ZDT1())
        
    def test_ZDT2(self):
        self._run_test(ZDT2())
        
    def test_ZDT3(self):
        self._run_test(ZDT3())
        
    def test_ZDT4(self):
        self._run_test(ZDT4())
        
    def test_ZDT5(self):
        self._run_test(ZDT5())
        
    def test_ZDT6(self):
        self._run_test(ZDT6())
        
    def _run_test(self, problem):
        if hasattr(problem, "random"):
            solution = problem.random()
        else:
            solution = RandomGenerator().generate(problem)
            
        problem.evaluate(solution)