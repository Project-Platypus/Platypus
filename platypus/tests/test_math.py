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

from .._math import (SingularError, add, choose, dot, lsolve, magnitude,
                     multiply, point_line_dist, subtract)


def test_dot():
    assert 0 == dot([0, 0, 0], [1, 1, 1])
    assert 1 == dot([1, 0, 1], [0, 1, 1])
    assert 2 == dot([1, 0, 1], [1, 1, 1])

def test_add():
    assert [1, 2, 3] == add([1, 1, 1], [0, 1, 2])

def test_subtract():
    assert [1, 0, -1] == subtract([1, 1, 1], [0, 1, 2])

def test_multiply():
    assert [0, 0, 0] == multiply(0, [1, 1, 1])
    assert [.5, .5, .5] == multiply(.5, [1, 1, 1])

def test_magnitude():
    assert 0 == magnitude([0, 0, 0])
    assert pytest.approx(1, abs=0.001) == magnitude([0.577, 0.577, 0.577])

def test_distance1():
    line = [1, 1, 1]
    point = [2, 2, 2]
    assert 0.0 == point_line_dist(point, line)

def test_distance2():
    line = [1, 1, 1]
    point = [0, 0, 1]
    assert pytest.approx(0.816, abs=0.001) == point_line_dist(point, line)

def test_distance3():
    line = [1, 0, 0]
    point = [0, 0, 1]
    assert 1.0 == point_line_dist(point, line)

def test_lsolve1():
    A = [[1, 0], [0, 1]]
    b = [1, 1]
    assert [1, 1] == lsolve(A, b)

def test_lsolve2():
    A = [[0.7, 0.3, 0.0], [0.1, 0.1, 0.7], [0.2, 0.1, 0.9]]
    b = [0.2, 0.3, 0.5]
    x = lsolve(A, b)
    assert pytest.approx(0.965, abs=0.001) == x[0]
    assert pytest.approx(-1.586, abs=0.001) == x[1]
    assert pytest.approx(0.517, abs=0.001) == x[2]

def test_lsolve_singular():
    A = [[0.5, 0.5], [0.5, 0.5]]
    b = [1, 1]
    with pytest.raises(SingularError):
        lsolve(A, b)

def test_choose():
    assert 1 == choose(0, 0)
    assert 1 == choose(1, 0)
    assert 1 == choose(5, 0)
    assert 1 == choose(1, 1)
    assert 5 == choose(5, 1)
    assert 5 == choose(5, 4)
    assert 1 == choose(5, 5)
