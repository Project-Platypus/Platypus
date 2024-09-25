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

from ..types import (Binary, Integer, Permutation, Real, Subset, bin2gray,
                     bin2int, gray2bin, int2bin)
from ._utils import assertBinEqual


@pytest.fixture
def nsamples():
    return 100

elements = [("foo", 5), ("bar", 2), "hello", "world"]

test_cases = [
    pytest.param(Real(0.0, 5.0), lambda x: isinstance(x, float) and x >= 0.0 and x <= 5.0, id="Real"),
    pytest.param(Integer(0, 5), lambda x: isinstance(x, int) and x >= 0 and x <= 5, id="Integer"),
    pytest.param(Binary(5), lambda x: isinstance(x, list) and len(x) == 5 and all([isinstance(v, bool) for v in x]), id="Binary"),
    pytest.param(Permutation(range(5)), lambda x: isinstance(x, list) and len(x) == 5 and all([v in range(5) for v in x]), id="Permutation-Int"),
    pytest.param(Permutation(elements), lambda x: isinstance(x, list) and len(x) == 4 and all([v in elements for v in x]), id="Permutation-Elements"),
    pytest.param(Subset(range(10), 2), lambda x: isinstance(x, list) and len(x) == 2 and all([v in range(10) for v in x]), id="Subset-Int"),
    pytest.param(Subset(elements, 2), lambda x: isinstance(x, list) and len(x) == 2 and all([v in elements for v in x]), id="Subset-Elements"),
]

@pytest.mark.parametrize("type,validator", test_cases)
def test(type, validator, nsamples):
    for i in range(nsamples):
        x = type.rand()
        decoded = type.decode(x)

        assert validator(decoded)
        assert x == type.encode(decoded)

def test_real_bounds():
    type = Real(0.0, 5.0)
    assert 0.0 == type.min_value
    assert 5.0 == type.max_value

def test_int_bounds():
    type = Integer(0, 5)
    assert 3 == type.nbits
    assert 0 == type.min_value
    assert 5 == type.max_value

BINARY_ENCODINGS = {
    0: {"binary": (0, 0, 0, 0), "gray": (0, 0, 0, 0)},
    1: {"binary": (0, 0, 0, 1), "gray": (0, 0, 0, 1)},
    2: {"binary": (0, 0, 1, 0), "gray": (0, 0, 1, 1)},
    3: {"binary": (0, 0, 1, 1), "gray": (0, 0, 1, 0)},
    4: {"binary": (0, 1, 0, 0), "gray": (0, 1, 1, 0)},
    5: {"binary": (0, 1, 0, 1), "gray": (0, 1, 1, 1)},
    6: {"binary": (0, 1, 1, 0), "gray": (0, 1, 0, 1)},
    7: {"binary": (0, 1, 1, 1), "gray": (0, 1, 0, 0)},
    8: {"binary": (1, 0, 0, 0), "gray": (1, 1, 0, 0)},
    9: {"binary": (1, 0, 0, 1), "gray": (1, 1, 0, 1)},
    10: {"binary": (1, 0, 1, 0), "gray": (1, 1, 1, 1)},
    11: {"binary": (1, 0, 1, 1), "gray": (1, 1, 1, 0)},
    12: {"binary": (1, 1, 0, 0), "gray": (1, 0, 1, 0)},
    13: {"binary": (1, 1, 0, 1), "gray": (1, 0, 1, 1)},
    14: {"binary": (1, 1, 1, 0), "gray": (1, 0, 0, 1)},
    15: {"binary": (1, 1, 1, 1), "gray": (1, 0, 0, 0)}}

def test_int2bin():
    assertBinEqual([], int2bin(0, 0))
    assertBinEqual([0], int2bin(0, 1))

    for i in BINARY_ENCODINGS.keys():
        assertBinEqual(BINARY_ENCODINGS[i]["binary"], int2bin(i, 4))

def test_bin2int():
    assert 0 == bin2int([])
    assert 0 == bin2int([0])

    for i in BINARY_ENCODINGS.keys():
        assert i == bin2int(BINARY_ENCODINGS[i]["binary"])

def test_bin2gray():
    for i in BINARY_ENCODINGS.keys():
        assertBinEqual(BINARY_ENCODINGS[i]["gray"], bin2gray(int2bin(i, 4)))

def test_gray2bin():
    for i in BINARY_ENCODINGS.keys():
        assertBinEqual(BINARY_ENCODINGS[i]["binary"], gray2bin(BINARY_ENCODINGS[i]["gray"]))
