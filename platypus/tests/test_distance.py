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
import pytest

from ..distance import DistanceMatrix, euclidean_dist, manhattan_dist
from ._utils import createSolution


def test_euclidean():
    assert 0.0 == euclidean_dist([1, 1], [1, 1])
    assert pytest.approx(1.414, abs=0.001) == euclidean_dist([0, 0], [1, 1])
    assert pytest.approx(1.414, abs=0.001) == euclidean_dist([1, 1], [0, 0])

def test_manhattan():
    assert 0.0 == manhattan_dist([1, 1], [1, 1])
    assert pytest.approx(2.0, abs=0.001) == manhattan_dist([0, 0], [1, 1])
    assert pytest.approx(2.0, abs=0.001) == manhattan_dist([1, 1], [0, 0])

def test_distance_matrix():
    solutions = [createSolution(0, 1), createSolution(0.5, 0.5), createSolution(0.75, 0.25), createSolution(1, 0)]
    matrix = DistanceMatrix(solutions)

    assert pytest.approx(0.353, abs=0.001) == matrix[1, 2]
    assert pytest.approx(0.353, abs=0.001) == matrix[2, 1]
    assert pytest.approx(0.353, abs=0.001) == matrix.kth_distance(2, 0)
    assert pytest.approx(0.353, abs=0.001) == matrix.kth_distance(1, 0)

    assert 2 == matrix.find_most_crowded()
    matrix.remove_point(2)
    assert 1 == matrix.find_most_crowded()
