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
from __future__ import absolute_import, division, print_function

import math
import functools
from .core import Solution, Problem, Indicator, normalize, POSITIVE_INFINITY
from .tools import euclidean_dist

def normalized_euclidean_dist(x, y):
    return euclidean_dist(x.normalized_objectives, y.normalized_objectives)

def manhattan_dist(x, y):
    if isinstance(x, Solution):
        x = x.objectives
    if isinstance(y, Solution):
        y = y.objectives

    return math.sqrt(sum([abs(x[i]-y[i]) for i in range(len(x))]))

def distance_to_nearest(solution, set):
    if len(set) == 0:
        return POSITIVE_INFINITY
    
    return min([normalized_euclidean_dist(solution, s) for s in set])

class GenerationalDistance(Indicator):
    
    def __init__(self, reference_set, d = 2.0):
        super(GenerationalDistance, self).__init__()
        self.reference_set = [s for s in reference_set if s.constraint_violation==0.0]
        self.d = d
        self.minimum, self.maximum = normalize(reference_set)

    def calculate(self, set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        
        if len(feasible) == 0:
            return POSITIVE_INFINITY
        
        normalize(feasible, self.minimum, self.maximum)
        return math.pow(sum([math.pow(distance_to_nearest(s, self.reference_set), self.d) for s in feasible]), 1.0 / self.d) / len(feasible)

class InvertedGenerationalDistance(Indicator):
    
    def __init__(self, reference_set, d = 1.0):
        super(InvertedGenerationalDistance, self).__init__()
        self.reference_set = [s for s in reference_set if s.constraint_violation==0.0]
        self.d = d
        self.minimum, self.maximum = normalize(reference_set)

    def calculate(self, set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        normalize(feasible, self.minimum, self.maximum)
        return math.pow(sum([math.pow(distance_to_nearest(s, feasible), self.d) for s in self.reference_set]), 1.0 / self.d) / len(self.reference_set)

class EpsilonIndicator(Indicator):
    
    def __init__(self, reference_set):
        super(EpsilonIndicator, self).__init__()
        self.reference_set = [s for s in reference_set if s.constraint_violation==0.0]
        self.minimum, self.maximum = normalize(reference_set)

    def calculate(self, set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        
        if len(feasible) == 0:
            return POSITIVE_INFINITY
        
        normalize(feasible, self.minimum, self.maximum)
        return max([min([max([s2.normalized_objectives[k] - s1.normalized_objectives[k] for k in range(s2.problem.nobjs)]) for s2 in set]) for s1 in self.reference_set])
    
class Spacing(Indicator):
    
    def __init__(self):
        super(Spacing, self).__init__()
        
    def calculate(self, set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        distances = []
        
        if len(feasible) < 2:
            return 0.0
    
        for s1 in feasible:
            distances.append(min([manhattan_dist(s1, s2) for s2 in feasible if s1 != s2]))
            
        avg_distance = sum(distances) / len(feasible)
        return math.sqrt(sum([math.pow(d - avg_distance, 2.0) for d in distances]) / (len(feasible)-1))
    
class Hypervolume(Indicator):
    
    def __init__(self, reference_set=None, minimum=None, maximum=None):
        if reference_set is not None:
            if minimum is not None or maximum is not None:
                raise ValueError("minimum and maximum must not be specified if reference_set is defined")
            self.reference_set = [s for s in reference_set if s.constraint_violation==0.0]
            self.minimum, self.maximum = normalize(reference_set)
        else:
            if minimum is None or maximum is None:
                raise ValueError("minimum and maximum must be specified when no reference_set is defined")
            self.minimum, self.maximum = minimum, maximum 
            
    def invert(self, solution):
        for i in range(solution.problem.nobjs):
            if solution.problem.directions[i] == Problem.MINIMIZE:
                solution.normalized_objectives[i] = 1.0 - max(0.0, min(1.0, solution.normalized_objectives[i]))
    
    def dominates(self, solution1, solution2, nobjs):
        better = False
        worse = False
        
        for i in range(nobjs):
            if solution1.normalized_objectives[i] > solution2.normalized_objectives[i]:
                better = True
            else:
                worse = True
                break
            
        return not worse and better
    
    def swap(self, solutions, i, j):
        solutions[i], solutions[j] = solutions[j], solutions[i]
        
    def filter_nondominated(self, solutions, nsols, nobjs):
        i = 0
        n = nsols
        
        while i < n:
            j = i + 1
            
            while j < n:
                if self.dominates(solutions[i], solutions[j], nobjs):
                    n -= 1
                    self.swap(solutions, j, n)
                elif self.dominates(solutions[j], solutions[i], nobjs):
                    n -= 1
                    self.swap(solutions, i, n)
                    i -= 1
                    break
                else:
                    j += 1
                
            i += 1
            
        return n
    
    def surface_unchanged_to(self, solutions, nsols, obj):
        return min([solutions[i].normalized_objectives[obj] for i in range(nsols)])
    
    def reduce_set(self, solutions, nsols, obj, threshold):
        i = 0
        n = nsols
        
        while i < n:
            if solutions[i].normalized_objectives[obj] <= threshold:
                n -= 1
                self.swap(solutions, i, n)
            i += 1
            
        return n
    
    def calc_internal(self, solutions, nsols, nobjs):
        volume = 0.0
        distance = 0.0
        n = nsols
        
        while n > 0:
            nnondom = self.filter_nondominated(solutions, n, nobjs-1)
            
            if nobjs < 3:
                temp_volume = solutions[0].normalized_objectives[0]
            else:
                temp_volume = self.calc_internal(solutions, nnondom, nobjs-1)
                
            temp_distance = self.surface_unchanged_to(solutions, n, nobjs-1)
            volume += temp_volume * (temp_distance - distance)
            distance = temp_distance
            n = self.reduce_set(solutions, n, nobjs-1, distance)
            
        return volume
    
    def calculate(self, set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        normalize(feasible, self.minimum, self.maximum)
        feasible = [s for s in feasible if all([o <= 1.0 for o in s.normalized_objectives])]
            
        if len(feasible) == 0:
            return 0.0
            
        for s in feasible:
            self.invert(s)
                
        return self.calc_internal(feasible, len(feasible), set[0].problem.nobjs)
