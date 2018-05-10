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
from ..problems import DTLZ2
from ..algorithms import *
from ..weights import *

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

    def test_MOEAD_random_weights(self):
        pickle.dumps(MOEAD(self.problem))

    def test_MOEAD_normal_boundary_weights(self):
        pickle.dumps(MOEAD(self.problem, weight_generator=normal_boundary_weights, divisions_outer=24))
        
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
        self.post_checks = lambda : True
    
    def test_NSGAII(self):
        self.algorithm = NSGAII(self.problem)
        self._run_test()
    
    def test_NSGAIII(self):
        self.algorithm = NSGAIII(self.problem, divisions_outer=24)
        self._run_test()
        
    def test_CMAES(self):
        self.algorithm = CMAES(self.problem)
        self._run_test()
        
    def test_GDE3(self):
        self.algorithm = GDE3(self.problem)
        self._run_test()
        
    def test_IBEA(self):
        self.algorithm = IBEA(self.problem)
        self._run_test()
        
    def test_MOEAD_default(self):
        self.algorithm = MOEAD(self.problem)
        self.post_checks = lambda : self.assertEqual(100, self.algorithm.population_size)
        self._run_test()

    def test_MOEAD_random_weights(self):
        self.algorithm = MOEAD(self.problem, population_size=50)
        self.post_checks = lambda : self.assertEqual(50, self.algorithm.population_size)
        self._run_test()

    def test_MOEAD_normal_boundary_weights(self):
        self.algorithm = MOEAD(self.problem, weight_generator=normal_boundary_weights, divisions_outer=24)
        self._run_test()
        
    def test_MOEAD_pbi(self):
        self.algorithm = MOEAD(self.problem, scalarizing_function=functools.partial(pbi, theta=0.5))
        self._run_test()
        
    def test_OMOPSO(self):
        self.algorithm = OMOPSO(self.problem, epsilons=[0.01])
        self._run_test()
        
    def test_SMPSO(self):
        self.algorithm = SMPSO(self.problem)
        self._run_test()
        
    def test_SPEA2(self):
        self.algorithm = SPEA2(self.problem)
        self._run_test()
        
    def test_EpsMOEA(self):
        self.algorithm = EpsMOEA(self.problem, epsilons=[0.01])
        self._run_test()
        
    def _run_test(self):
        self.algorithm.run(100)
        self.post_checks()

class TestMaximizationGuard(unittest.TestCase):
    
    def setUp(self):
        self.problem = Problem(1, 1)
        self.problem.directions[:] = Problem.MAXIMIZE
        
    def test_MOEAD(self):
        with self.assertRaises(PlatypusError):
            MOEAD(self.problem)
            
    def test_NSGAIII(self):
        with self.assertRaises(PlatypusError):
            NSGAIII(self.problem, divisions_outer = 24)