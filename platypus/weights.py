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
from __future__ import absolute_import, division, print_function

import math
import copy
import random
from .core import POSITIVE_INFINITY

def chebyshev(values, weights, min_weight=0.0001):
    return max([max(weights[i], min_weight) * values[i] for i in range(len(values))])

def random_weights(size, nobjs):
    weights = []
    
    if nobjs == 2:
        weights = [[1, 0], [0, 1]]
        weights.extend([(i/(size-1.0), 1.0-i/(size-1.0)) for i in range(1, size-1)])
    else:
        # generate candidate weights
        candidate_weights = []
        
        for i in range(size*50):
            random_values = [random.uniform(0.0, 1.0) for _ in range(nobjs)]
            candidate_weights.append([x/sum(random_values) for x in random_values])
        
        # add weights for the corners
        for i in range(nobjs):
            weights.append([0]*i + [1] + [0]*(nobjs-i-1))
            
        # iteratively fill in the remaining weights by finding the candidate
        # weight with the largest distance from the assigned weights
        while len(weights) < size:
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