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

import sys
import math
import copy
import random
from .core import POSITIVE_INFINITY

# Scalarizing functions used to compute the fitness of a solution with multiple
# objectives.

def chebyshev(solution, ideal_point, weights, min_weight=0.0001):
    """Chebyshev (Tchebycheff) fitness of a solution with multiple objectives.
    
    This function is designed to only work with minimized objectives.
    
    Parameters
    ----------
    solution : Solution
        The solution.
    ideal_point : list of float
        The ideal point.
    weights : list of float
        The weights.
    min_weight : float
        The minimum weight allowed.
    """
    nobjs = solution.problem.nobjs
    objs = solution.objectives
    return max([max(weights[i], min_weight) * (objs[i]-ideal_point[i]) for i in range(nobjs)])

def pbi(solution, ideal_point, weights, theta):
    """Penalty-based boundary intersection fitness of a solution with multiple objectives.
    
    Requires numpy.  This function is designed to only work with minimized
    objectives.
    
    Callers need to set the theta value by using
        functools.partial(pbi, theta=0.5)
    
    Parameters
    ----------
    solution : Solution
        The solution.
    ideal_point: list of float
        The ideal point.
    weights : list of float
        The weights.
    theta : float
        The theta value.
    """
    try:
        import numpy as np
    except:
        print("The pbi function requires numpy.", file=sys.stderr)
        raise

    w = np.array(weights)
    z_star = np.array(ideal_point)
    F = np.array(solution.objectives)

    d1 = np.linalg.norm(np.dot((F - z_star), w)) / np.linalg.norm(w)
    d2 = np.linalg.norm(F - (z_star + d1 * w))

    return (d1 + theta * d2).tolist()

# Implementation Note: The first argument to weight vector generators should be
# nobjs, the number of objectives.  The remaining arguments should be provided
# by the algorithm constructor.

def random_weights(nobjs, population_size):
    """Returns a set of randomly-generated but uniformly distributed weights.
    
    Simply producing N randomly-generated weights does not necessarily produce
    uniformly-distributed weights.  To help produce more uniformly-distributed
    weights, this method picks weights from a large collection of randomly-
    generated weights such that the distances between weights is maximized.
    
    Parameters
    ----------
    nobjs : int
        The number of objectives.
    population_size : int
        The number of weights to generate.
    """
    
    weights = []
    
    if nobjs == 2:
        weights = [[1, 0], [0, 1]]
        weights.extend([(i/(population_size-1.0), 1.0-i/(population_size-1.0)) for i in range(1, population_size-1)])
    else:
        # generate candidate weights
        candidate_weights = []
        
        for i in range(population_size*50):
            random_values = [random.uniform(0.0, 1.0) for _ in range(nobjs)]
            candidate_weights.append([x/sum(random_values) for x in random_values])
        
        # add weights for the corners
        for i in range(nobjs):
            weights.append([0]*i + [1] + [0]*(nobjs-i-1))
            
        # iteratively fill in the remaining weights by finding the candidate
        # weight with the largest distance from the assigned weights
        while len(weights) < population_size:
            max_index = -1
            max_distance = -POSITIVE_INFINITY
            
            for i in range(len(candidate_weights)):
                distance = POSITIVE_INFINITY
                
                for j in range(len(weights)):
                    temp = math.sqrt(sum([math.pow(candidate_weights[i][k]-weights[j][k], 2.0) for k in range(nobjs)]))
                    distance = min(distance, temp)
                    
                if distance > max_distance:
                    max_index = i
                    max_distance = distance
                    
            weights.append(candidate_weights[max_index])
            del candidate_weights[max_index]
            
    return weights

def normal_boundary_weights(nobjs, divisions_outer, divisions_inner=0):
    """Returns weights generated by the normal boundary method.
    
    The weights produced by this method are uniformly distributed on the
    hyperplane intersecting
    
        [(1, 0, ..., 0), (0, 1, ..., 0), ..., (0, 0, ..., 1)].
        
    Parameters
    ----------
    nobjs : int
        The number of objectives.
    divisions_outer : int
        The number of divisions along the outer set of weights.
    divisions_inner : int (optional)
        The number of divisions along the inner set of weights.
    """
    
    def generate_recursive(weights, weight, left, total, index):
        if index == nobjs - 1:
            weight[index] = float(left) / float(total)
            weights.append(copy.copy(weight))
        else:
            for i in range(left+1):
                weight[index] = float(i) / float(total)
                generate_recursive(weights, weight, left-i, total, index+1)
    
    def generate_weights(divisions):
        weights = []
        generate_recursive(weights, [0.0]*nobjs, divisions, divisions, 0)
        return weights
        
    weights = generate_weights(divisions_outer)
    
    if divisions_inner > 0:
        inner_weights = generate_weights(divisions_inner)
        
        for i in range(len(inner_weights)):
            weight = inner_weights[i]
            
            for j in range(len(weight)):
                weight[j] = (1.0 / nobjs + weight[j]) / 2.0
                
            weights.append(weight)
        
    return weights