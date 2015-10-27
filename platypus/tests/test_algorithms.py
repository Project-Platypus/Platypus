# Copyright 2015 David Hadka
#
# This file is part of Platypus.
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
from ..problems import DTLZ2
from ..algorithms import NSGAII
from ..operators import GAOperator, SBX, PM

class TestPickling(unittest.TestCase):
    
    def test_NSGAII(self):
        self.skipTest("in development")
        problem = DTLZ2()
        algorithm = NSGAII(problem, variator=GAOperator(SBX(), PM()))
        pickle.dumps(algorithm)