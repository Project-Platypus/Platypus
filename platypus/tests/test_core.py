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
import copy
import random
import unittest
from ..core import Constraint, Problem, Solution, ParetoDominance, Archive, \
        nondominated_sort, nondominated_truncate, nondominated_prune, \
        POSITIVE_INFINITY, nondominated_split, truncate_fitness, normalize, \
        EpsilonBoxArchive

def createSolution(*args):
    problem = Problem(0, len(args))
    solution = Solution(problem)
    solution.objectives[:] = [float(x) for x in args]
    return solution

class TestSolution(unittest.TestCase):
    
    def test_deepcopy(self):
        orig = createSolution(4, 5)
        orig.constraint_violation = 2
        orig.evaluated = True
        
        clone = copy.deepcopy(orig)
        
        self.assertEqual(orig.problem, clone.problem)
        self.assertEqual(4, clone.objectives[0])
        self.assertEqual(5, clone.objectives[1])
        self.assertEqual(2, clone.constraint_violation)
        self.assertEqual(True, clone.evaluated)
        

class TestConstraint(unittest.TestCase):
    
    def test_eq(self):
        constraint = Constraint("==0")
        self.assertEqual(0.0, constraint(0.0))
        self.assertNotEqual(0.0, constraint(1.0))
        self.assertNotEqual(0.0, constraint(-1.0))
        
        constraint = Constraint("==5")
        self.assertEqual(0.0, constraint(5.0))
        self.assertNotEqual(0.0, constraint(-5.0))
        self.assertNotEqual(0.0, constraint(10.0))
        
    def test_lte(self):
        constraint = Constraint("<=0")
        self.assertEqual(0.0, constraint(0.0))
        self.assertEqual(0.0, constraint(-1.0))
        self.assertNotEqual(0.0, constraint(1.0))
        
    def test_gte(self):
        constraint = Constraint(">=0")
        self.assertEqual(0.0, constraint(0.0))
        self.assertEqual(0.0, constraint(1.0))
        self.assertNotEqual(0.0, constraint(-1.0))
        
    def test_lt(self):
        constraint = Constraint("<0")
        self.assertNotEqual(0.0, constraint(0.0))
        self.assertNotEqual(0.0, constraint(1.0))
        self.assertEqual(0.0, constraint(-1.0))
        
    def test_gt(self):
        constraint = Constraint(">0")
        self.assertNotEqual(0.0, constraint(0.0))
        self.assertEqual(0.0, constraint(1.0))
        self.assertNotEqual(0.0, constraint(-1.0))
        
    def test_neq(self):
        constraint = Constraint("!=0")
        self.assertNotEqual(0.0, constraint(0.0))
        self.assertEqual(0.0, constraint(1.0))
        self.assertEqual(0.0, constraint(-1.0))
        
class TestParetoDominance(unittest.TestCase):
    
    def test_dominance(self):
        dominance = ParetoDominance()
        s1 = createSolution(0.0, 0.0)
        s2 = createSolution(1.0, 1.0)
        s3 = createSolution(0.0, 1.0)
        
        self.assertEqual(-1, dominance.compare(s1, s2))
        self.assertEqual(1, dominance.compare(s2, s1))
        self.assertEqual(-1, dominance.compare(s1, s3))
        self.assertEqual(1, dominance.compare(s3, s1))
        
    def test_nondominance(self):
        dominance = ParetoDominance()
        s1 = createSolution(0.0, 1.0)
        s2 = createSolution(0.5, 0.5)
        s3 = createSolution(1.0, 0.0)
        
        self.assertEqual(0, dominance.compare(s1, s2))
        self.assertEqual(0, dominance.compare(s2, s1))
        self.assertEqual(0, dominance.compare(s2, s3))
        self.assertEqual(0, dominance.compare(s3, s2))
        self.assertEqual(0, dominance.compare(s1, s3))
        self.assertEqual(0, dominance.compare(s3, s1))
        
class TestArchive(unittest.TestCase):
    
    def test_dominance(self):
        s1 = createSolution(1.0, 1.0)
        s2 = createSolution(0.0, 0.0)
        s3 = createSolution(0.0, 1.0)
        
        archive = Archive(ParetoDominance())
        archive += [s1, s2, s3]
        
        self.assertEqual(1, len(archive))
        self.assertEqual(s2, archive[0])
        
    def test_nondominance(self):
        s1 = createSolution(0.0, 1.0)
        s2 = createSolution(0.5, 0.5)
        s3 = createSolution(1.0, 0.0)
        
        archive = Archive(ParetoDominance())
        archive += [s1, s2, s3]
        
        self.assertEqual(3, len(archive))
        
class TestNondominatedSort(unittest.TestCase):
    
    def setUp(self):
        self.s1 = createSolution(0.0, 1.0)
        self.s2 = createSolution(0.5, 0.5)
        self.s3 = createSolution(1.0, 0.0)
        self.s4 = createSolution(0.75, 0.75)
        self.s5 = createSolution(1.0, 1.0)
        
        self.population = [self.s1, self.s2, self.s3, self.s4, self.s5]
        random.shuffle(self.population)
        
    def test_ranking(self):
        nondominated_sort(self.population)
        
        self.assertEqual(0, self.s1.rank)
        self.assertEqual(0, self.s2.rank)
        self.assertEqual(0, self.s3.rank)
        self.assertEqual(1, self.s4.rank)
        self.assertEqual(2, self.s5.rank)
        
    def test_crowding(self):
        nondominated_sort(self.population)
        
        self.assertEqual(POSITIVE_INFINITY, self.s1.crowding_distance)
        self.assertEqual(2, self.s2.crowding_distance)
        self.assertEqual(POSITIVE_INFINITY, self.s3.crowding_distance)
        self.assertEqual(POSITIVE_INFINITY, self.s4.crowding_distance)
        self.assertEqual(POSITIVE_INFINITY, self.s5.crowding_distance)
        
    def test_split2(self):
        nondominated_sort(self.population)
        (first, second) = nondominated_split(self.population, 2)
        
        self.assertEqual(0, len(first))
        self.assertEqual(3, len(second))
        self.assertIn(self.s1, second)
        self.assertIn(self.s2, second)
        self.assertIn(self.s3, second)
        
    def test_split3(self):
        nondominated_sort(self.population)
        (first, second) = nondominated_split(self.population, 3)
        
        self.assertEqual(3, len(first))
        self.assertEqual(0, len(second))
        self.assertIn(self.s1, first)
        self.assertIn(self.s2, first)
        self.assertIn(self.s3, first)
        
    def test_split4(self):
        nondominated_sort(self.population)
        (first, second) = nondominated_split(self.population, 4)
        
        self.assertEqual(4, len(first))
        self.assertEqual(0, len(second))
        self.assertIn(self.s1, first)
        self.assertIn(self.s2, first)
        self.assertIn(self.s3, first)
        self.assertIn(self.s4, first)
        
    def test_truncate2(self):
        nondominated_sort(self.population)
        result = nondominated_truncate(self.population, 2)
        
        self.assertEqual(2, len(result))
        self.assertIn(self.s1, result)
        self.assertIn(self.s3, result)
        
    def test_truncate4(self):
        nondominated_sort(self.population)
        result = nondominated_truncate(self.population, 4)
        
        self.assertEqual(4, len(result))
        self.assertIn(self.s1, result)
        self.assertIn(self.s2, result)
        self.assertIn(self.s3, result)
        self.assertIn(self.s4, result)
        
    def test_prune2(self):
        nondominated_sort(self.population)
        result = nondominated_prune(self.population, 2)
        
        self.assertEqual(2, len(result))
        self.assertIn(self.s1, result)
        self.assertIn(self.s3, result)
        
    def test_prune4(self):
        nondominated_sort(self.population)
        result = nondominated_prune(self.population, 4)
        
        self.assertEqual(4, len(result))
        self.assertIn(self.s1, result)
        self.assertIn(self.s2, result)
        self.assertIn(self.s3, result)
        self.assertIn(self.s4, result)
        
    def test_truncate_fitness_max(self):
        self.s1.fitness = 1
        self.s2.fitness = 5
        self.s3.fitness = 3
        self.s4.fitness = 4
        self.s5.fitness = 2
        
        result = truncate_fitness(self.population, 3)
        
        self.assertEqual(3, len(result))
        self.assertIn(self.s2, result)
        self.assertIn(self.s4, result)
        self.assertIn(self.s3, result)
        
    def test_truncate_fitness_min(self):
        self.s1.fitness = 1
        self.s2.fitness = 5
        self.s3.fitness = 3
        self.s4.fitness = 4
        self.s5.fitness = 2
        
        result = truncate_fitness(self.population, 3, larger_preferred=False)
        
        self.assertEqual(3, len(result))
        self.assertIn(self.s1, result)
        self.assertIn(self.s5, result)
        self.assertIn(self.s3, result)
        
class TestNormalize(unittest.TestCase):
    
    def test_normalize(self):
        s1 = createSolution(0, 2)
        s2 = createSolution(2, 3)
        s3 = createSolution(1, 1)
        solutions = [s1, s2, s3]
        
        normalize(solutions)
        
        self.assertEqual([0.0, 0.5], s1.normalized_objectives)
        self.assertEqual([1.0, 1.0], s2.normalized_objectives)
        self.assertEqual([0.5, 0.0], s3.normalized_objectives)
        
class TestEpsilonBoxArchive(unittest.TestCase):
    
    def test_improvements(self):
        s1 = createSolution(0.25, 0.25)
        s2 = createSolution(0.1, 0.1)
        s3 = createSolution(0.245, 0.245)
        s4 = createSolution(0.1, 0.5)
        s5 = createSolution(0.5, 0.5)
        s6 = createSolution(0.0, 0.0)
        
        archive = EpsilonBoxArchive([0.1])
        
        archive.extend([s1, s2, s3, s4, s5, s6])
        self.assertEqual(2, archive.improvements)