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

from ..filters import (group, matches, objective_value_at_index,
                       objectives_key, truncate, unique)
from ._utils import createSolution

s1 = createSolution(0.0, 1.0)
s2 = createSolution(1.0, 0.0)
s3 = createSolution(0.0, 1.0)

def iterator(args):
    return iter(args)

def generator(args):
    return (x for x in args)

def view(args):
    return {x: x for x in args}.keys()

def truncate_2nd_obj(solutions):
    return truncate(solutions, 1, key=objective_value_at_index(1))

def matches_2nd_obj(solutions):
    return matches(solutions, 1.0, key=objective_value_at_index(1))

def test_objectives_key():
    assert (0.0, 1.0) == objectives_key(createSolution(0.0, 1.0))

def test_objective_value_at_index():
    s = createSolution(0.0, 1.0)
    assert 0.0 == objective_value_at_index(0)(s)
    assert 1.0 == objective_value_at_index(1)(s)

@pytest.mark.parametrize("source", [list, iterator, generator, view])
@pytest.mark.parametrize("filter,expected", [
    (unique, []),
    (group, {}),
    (truncate_2nd_obj, []),
    (matches_2nd_obj, [])])
def test_empty(source, filter, expected):
    assert expected == filter(source([]))

@pytest.mark.parametrize("source", [list, iterator, generator, view])
@pytest.mark.parametrize("filter,expected", [
    (unique, [s1]),
    (group, {(0.0, 1.0): [s1]}),
    (truncate_2nd_obj, [s1]),
    (matches_2nd_obj, [s1])])
def test_single_item(source, filter, expected):
    assert expected == filter(source([s1]))

@pytest.mark.parametrize("source", [list, iterator, generator, view])
@pytest.mark.parametrize("filter,expected", [
    (unique, [s1, s2]),
    (group, {(0.0, 1.0): [s1, s3], (1.0, 0.0): [s2]}),
    (truncate_2nd_obj, [s2]),
    (matches_2nd_obj, [s1, s3])])
def test_multiple_items(source, filter, expected):
    assert expected == filter(source([s1, s2, s3]))
