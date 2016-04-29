# Copyright 2015-2016 David Hadka
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
from ..problems import DTLZ2
from ..algorithms import *

class TestPickling(unittest.TestCase):
    
    def setUp(self):
        self.problem = DTLZ2()
    
    def test_NSGAII(self):
        pickle.dumps(NSGAII(self.problem))
    
    def test_NSGAIII(self):
        pickle.dumps(NSGAIII(self.problem, divisions_outer=24))
        
    def test_CMAES(self):
        pickle.dumps(CMAES(self.problem))
        
    def test_GDE3(self):
        pickle.dumps(GDE3(self.problem))
        
    def test_IBEA(self):
        pickle.dumps(IBEA(self.problem))
        
    def test_MOEAD(self):
        pickle.dumps(MOEAD(self.problem))
        
    def test_OMOPSO(self):
        pickle.dumps(OMOPSO(self.problem, epsilons=[0.01]))
        
    def test_SMPSO(self):
        pickle.dumps(SMPSO(self.problem))
        
    def test_SPEA2(self):
        pickle.dumps(SPEA2(self.problem))
        
    def test_EpsMOEA(self):
        pickle.dumps(EpsMOEA(self.problem, epsilons=[0.01]))


class TestRunning(unittest.TestCase):
    
    def setUp(self):
        self.problem = DTLZ2()
    
    def test_NSGAII(self):
        self.algorithm = NSGAII(self.problem)
    
    def test_NSGAIII(self):
        self.algorithm = NSGAIII(self.problem, divisions_outer=24)
        
    def test_CMAES(self):
        self.algorithm = CMAES(self.problem)
        
    def test_GDE3(self):
        self.algorithm = GDE3(self.problem)
        
    def test_IBEA(self):
        self.algorithm = IBEA(self.problem)
        
    def test_MOEAD(self):
        self.algorithm = MOEAD(self.problem)
        
    def test_OMOPSO(self):
        self.algorithm = OMOPSO(self.problem, epsilons=[0.01])
        
    def test_SMPSO(self):
        self.algorithm = SMPSO(self.problem)
        
    def test_SPEA2(self):
        self.algorithm = SPEA2(self.problem)
        
    def test_EpsMOEA(self):
        self.algorithm = EpsMOEA(self.problem, epsilons=[0.01])

    def tearDown(self):
        self.algorithm.run(100)
