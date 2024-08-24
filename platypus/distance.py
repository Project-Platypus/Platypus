# Copyright 2015-2024 David Hadka
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

import math
from .core import Solution, POSITIVE_INFINITY


def euclidean_dist(x, y):
    """Computes the Euclidean distance between two points."""
    if isinstance(x, Solution):
        x = x.objectives
    if isinstance(y, Solution):
        y = y.objectives

    return math.sqrt(sum([math.pow(x[i]-y[i], 2.0) for i in range(len(x))]))

def normalized_euclidean_dist(x, y):
    return euclidean_dist(x.normalized_objectives, y.normalized_objectives)

def manhattan_dist(x, y):
    if isinstance(x, Solution):
        x = x.objectives
    if isinstance(y, Solution):
        y = y.objectives

    return sum([abs(x[i]-y[i]) for i in range(len(x))])

def distance_to_nearest(solution, set):
    if len(set) == 0:
        return POSITIVE_INFINITY

    return min([normalized_euclidean_dist(solution, s) for s in set])

class DistanceMatrix:
    """Maintains pairwise distances between solutions.

    The distance matrix, used by SPEA2, maintains the pairwise distances
    between solutions.  It also provides convenient routines to lookup the
    distance between any two solutions, find the most crowded solution, and
    remove a solution.
    """

    def __init__(self, solutions, distance_fun=euclidean_dist):
        super().__init__()
        self.distances = []

        for i in range(len(solutions)):
            distances_i = []
            for j in range(len(solutions)):
                if i != j:
                    distances_i.append((j, distance_fun(solutions[i], solutions[j])))

            self.distances.append(sorted(distances_i, key=lambda x: x[1]))

    def find_most_crowded(self):
        """Finds the most crowded solution.

        Returns the index of the most crowded solution, which is the solution
        with the smallest distance to the nearest neighbor.  Any ties are
        broken by looking at the next distant neighbor.
        """
        minimum_distance = POSITIVE_INFINITY
        minimum_index = -1

        for i in range(len(self.distances)):
            distances_i = self.distances[i]

            if distances_i[0][1] < minimum_distance:
                minimum_distance = distances_i[0][1]
                minimum_index = i
            elif distances_i[0][1] == minimum_distance:
                for j in range(len(distances_i)):
                    dist1 = distances_i[j][1]
                    dist2 = self.distances[minimum_index][j][1]

                    if dist1 < dist2:
                        minimum_index = i
                        break
                    if dist2 < dist1:
                        break

        return minimum_index

    def remove_point(self, index):
        """Removes the distance entries for the given solution.

        Parameters
        ----------
        index : int
            The index of the solution
        """
        del self.distances[index]

        for i in range(len(self.distances)):
            self.distances[i] = [(x if x < index else x-1, y) for (x, y) in self.distances[i] if x != index]

    def kth_distance(self, i, k):
        """Returns the distance to the k-th nearest neighbor.

        Parameters
        ----------
        i : int
            The index of the solution
        k : int
            Finds the k-th nearest neightbor distance
        """
        return self.distances[i][k][1]

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            if key[0] == key[1]:
                return 0.0
            else:
                for i, d in self.distances[key[0]]:
                    if i == key[1]:
                        return d
                raise ValueError("key not found")
        else:
            raise ValueError("key must be a tuple")
