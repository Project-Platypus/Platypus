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
import math
from .core import Solution, Problem, normalize, POSITIVE_INFINITY
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

def generational_distance(reference_set):
    """Returns a function for computing the generational distance.
    
    Returns a function for computing the generational distance of a set to
    a given reference_set.  Generational distance is the average Euclidean
    distance from each point in a population to the nearest point in a
    reference set.
    
    Parameters
    ----------
    reference_set : iterable
        The solutions comprising the reference set
    """
    reference_set = [s for s in reference_set if s.constraint_violation==0.0]
    minimum, maximum = normalize(reference_set)
    
    def calc(set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        
        if len(feasible) == 0:
            return POSITIVE_INFINITY
        
        normalize(feasible, minimum, maximum)
        return math.sqrt(sum([math.pow(distance_to_nearest(s, reference_set), 2.0) for s in feasible])) / len(feasible)
        
    return calc

def inverted_generational_distance(reference_set):
    """Returns a function for computing the inverted generational distance.
    
    Returns a function for computing the inverted generational distance of a
    set to a given reference_set.  Inverted generational distance is the
    average Euclidean distance from each point in a reference set to the
    nearest solution in the population.
    
    Parameters
    ----------
    reference_set : iterable
        The solutions comprising the reference set
    """
    reference_set = [s for s in reference_set if s.constraint_violation==0.0]
    minimum, maximum = normalize(reference_set)
    
    def calc(set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        normalize(feasible, minimum, maximum)
        return math.sqrt(sum([math.pow(distance_to_nearest(s, feasible), 2.0) for s in reference_set])) / len(reference_set)
        
    return calc
    
def epsilon_indicator(reference_set):
    """Returns a function for computing the additive epsilon indicator.
    
    Returns a function for computing the additive epsilon indicator of a
    set to a given reference_set.  Additive epsilon indicator measures the
    minimum distance that the population must be translated to dominate the
    reference set.
    
    Parameters
    ----------
    reference_set : iterable
        The solutions comprising the reference set
    """
    reference_set = [s for s in reference_set if s.constraint_violation==0.0]
    minimum, maximum = normalize(reference_set)
    
    def calc(set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        
        if len(feasible) == 0:
            return POSITIVE_INFINITY
        
        normalize(feasible, minimum, maximum)
        return max([min([max([s2.normalized_objectives[k] - s1.normalized_objectives[k] for k in range(s2.problem.nobjs)]) for s2 in set]) for s1 in reference_set])
        
    return calc
    
def spacing():
    def calc(set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        distances = []
        
        if len(feasible) < 2:
            return 0.0

        for s1 in feasible:
            distances.append(min([manhattan_dist(s1, s2) for s2 in feasible if s1 != s2]))
            
        avg_distance = sum(distances) / len(feasible)
        return math.sqrt(sum([math.pow(d - avg_distance, 2.0) for d in distances]) / (len(feasible)-1))
    
    return calc

def hypervolume(reference_set = None, minimum = None, maximum = None):
    if reference_set is not None:
        reference_set = [s for s in reference_set if s.constraint_violation==0.0]
        minimum, maximum = normalize(reference_set)
    
    def invert(solution):
        for i in range(solution.problem.nobjs):
            if solution.problem.directions[i] == Problem.MINIMIZE:
                solution.normalized_objectives[i] = 1.0 - max(0.0, min(1.0, solution.normalized_objectives[i]))
    
    def dominates(solution1, solution2, nobjs):
        better = False
        worse = False
        
        for i in range(nobjs):
            if solution1.normalized_objectives[i] > solution2.normalized_objectives[i]:
                better = True
            else:
                worse = True
                break
            
        return not worse and better
    
    def swap(solutions, i, j):
        solutions[i], solutions[j] = solutions[j], solutions[i]
        
    def filter_nondominated(solutions, nsols, nobjs):
        i = 0
        n = nsols
        
        while i < n:
            j = i + 1
            
            while j < n:
                if dominates(solutions[i], solutions[j], nobjs):
                    n -= 1
                    swap(solutions, j, n)
                elif dominates(solutions[j], solutions[i], nobjs):
                    n -= 1
                    swap(solutions, i, n)
                    i -= 1
                    break
                else:
                    j += 1
                
            i += 1
            
        return n
    
    def surface_unchanged_to(solutions, nsols, obj):
        return min([solutions[i].normalized_objectives[obj] for i in range(nsols)])
    
    def reduce_set(solutions, nsols, obj, threshold):
        i = 0
        n = nsols
        
        while i < n:
            if solutions[i].normalized_objectives[obj] <= threshold:
                n -= 1
                swap(solutions, i, n)
            i += 1
            
        return n
    
    def calc_internal(solutions, nsols, nobjs):
        volume = 0.0
        distance = 0.0
        n = nsols
        
        while n > 0:
            nnondom = filter_nondominated(solutions, n, nobjs-1)
            
            if nobjs < 3:
                temp_volume = solutions[0].normalized_objectives[0]
            else:
                temp_volume = calc_internal(solutions, nnondom, nobjs-1)
                
            temp_distance = surface_unchanged_to(solutions, n, nobjs-1)
            volume += temp_volume * (temp_distance - distance)
            distance = temp_distance
            n = reduce_set(solutions, n, nobjs-1, distance)
            
        return volume
    
    def calc(set):
        feasible = [s for s in set if s.constraint_violation==0.0]
        normalize(feasible, minimum, maximum)
        feasible = [s for s in feasible if all([o <= 1.0 for o in s.normalized_objectives])]
        
        if len(feasible) == 0:
            return 0.0
        
        for s in feasible:
            invert(s)
            
        return calc_internal(feasible, len(feasible), set[0].problem.nobjs)

    return calc   