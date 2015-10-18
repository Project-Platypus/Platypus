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
from .core import Solution, normalize, POSITIVE_INFINITY
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