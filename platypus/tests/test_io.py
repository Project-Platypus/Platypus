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
import tempfile
import pytest

from ..algorithms import NSGAII
from ..io import (load_json, load_objectives, load_state, save_json,
                  save_objectives, save_state)
from ..problems import CF1
from ._utils import createSolution, similar


def test_objectives():
    s1 = createSolution(0.0, 1.0)
    s2 = createSolution(1.0, 0.0)
    expected = [s1, s2]

    with tempfile.NamedTemporaryFile() as f:
        save_objectives(f.name, expected)
        actual = load_objectives(f.name, s1.problem)

    assert len(expected) == len(actual)

    for i in range(len(expected)):
        assert expected[i].objectives == actual[i].objectives

def test_json_solutions():
    s1 = createSolution(0.0, 1.0)
    s2 = createSolution(1.0, 0.0)
    expected = [s1, s2]

    with tempfile.NamedTemporaryFile() as f:
        save_json(f.name, expected)
        actual = load_json(f.name)

    assert len(expected) == len(actual)

    for i in range(len(expected)):
        assert actual[i].problem is not None
        similar(expected[i], actual[i])

def test_json_algorithm():
    problem = CF1()
    algorithm = NSGAII(problem)
    algorithm.run(1000)

    expected = algorithm.result

    with tempfile.NamedTemporaryFile() as f:
        save_json(f.name, algorithm)
        actual = load_json(f.name)

    assert len(expected) == len(actual)

    for i in range(len(expected)):
        assert actual[i].problem is not None
        similar(expected[i], actual[i])

@pytest.mark.parametrize("json", [False, True])
def test_state(json):
    problem = CF1()
    original = NSGAII(problem)

    with tempfile.NamedTemporaryFile() as f:
        save_state(f.name, original, json=json)

        original.run(1000)

        copy = load_state(f.name)
        copy.run(1000)

        assert original.nfe == copy.nfe

        expected = original.result
        actual = copy.result

        assert len(expected) == len(actual)

        for i in range(len(expected)):
            similar(expected[i], actual[i])
