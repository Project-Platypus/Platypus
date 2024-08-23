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
import unittest
from abc import ABCMeta, abstractmethod
from ..types import Real, Binary, Integer, Permutation, Subset, \
    bin2gray, bin2int, int2bin, gray2bin


class TypeTestCase(unittest.TestCase, metaclass=ABCMeta):

    @abstractmethod
    def createInstance(self):
        raise NotImplementedError()

    @abstractmethod
    def assertValidValue(self, val):
        raise NotImplementedError()

    def setUp(self):
        self.variable = self.createInstance()

    def test_rand(self):
        for i in range(100):
            val = self.variable.decode(self.variable.rand())
            self.assertValidValue(val)

    def test_encode_decode(self):
        for i in range(100):
            val = self.variable.rand()
            decoded = self.variable.decode(val)
            encoded = self.variable.encode(decoded)
            self.assertEqual(val, encoded)


class TestReal(TypeTestCase):

    def createInstance(self):
        return Real(0.0, 5.0)

    def assertValidValue(self, val):
        self.assertIsInstance(val, float)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 5.0)

    def test_init(self):
        self.assertEqual(0.0, self.variable.min_value)
        self.assertEqual(5.0, self.variable.max_value)


class TestBinary(TypeTestCase):

    def createInstance(self):
        return Binary(5)

    def assertValidValue(self, val):
        self.assertEqual(5, len(val))
        for v in val:
            self.assertIsInstance(v, bool)


class TestInteger(TypeTestCase):

    def createInstance(self):
        return Integer(0, 5)

    def assertValidValue(self, val):
        self.assertIsInstance(val, int)
        self.assertGreaterEqual(val, 0)
        self.assertLessEqual(val, 5)

    def test_init(self):
        self.assertEqual(3, self.variable.nbits)
        self.assertEqual(0, self.variable.min_value)
        self.assertEqual(5, self.variable.max_value)


class TestPermutationIntegers(TypeTestCase):

    def createInstance(self):
        return Permutation(range(5))

    def assertValidValue(self, val):
        self.assertEqual(5, len(val))
        for i in range(5):
            self.assertIn(i, val)


class TestPermutationElements(TypeTestCase):

    elements = [("foo", 5), ("bar", 2)]

    def createInstance(self):
        return Permutation(self.elements)

    def assertValidValue(self, val):
        self.assertEqual(2, len(val))
        for e in self.elements:
            self.assertIn(e, val)


class TestSubset(TypeTestCase):

    def createInstance(self):
        return Subset(range(10), 2)

    def assertValidValue(self, val):
        self.assertEqual(2, len(val))
        for v in val:
            self.assertIn(v, range(10))

class TestGrayCode(unittest.TestCase):

    EXPECTED = {
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
        15: {"binary": (1, 1, 1, 1), "gray": (1, 0, 0, 0)},
    }

    def assertBinEqual(self, b1, b2):
        self.assertEqual(len(b1), len(b2))

        for i in range(len(b1)):
            self.assertEqual(bool(b1[i]), bool(b2[i]))

    def test_int2bin(self):
        self.assertBinEqual([], int2bin(0, 0))
        self.assertBinEqual([0], int2bin(0, 1))

        for i in range(16):
            self.assertBinEqual(self.EXPECTED[i]["binary"], int2bin(i, 4))

    def test_bin2int(self):
        self.assertEqual(0, bin2int([]))
        self.assertEqual(0, bin2int([0]))

        for i in range(16):
            self.assertEqual(i, bin2int(self.EXPECTED[i]["binary"]))

    def test_bin2gray(self):
        for i in range(16):
            self.assertBinEqual(self.EXPECTED[i]["gray"], bin2gray(int2bin(i, 4)))

    def test_gray2bin(self):
        for i in range(16):
            self.assertBinEqual(self.EXPECTED[i]["binary"], gray2bin(self.EXPECTED[i]["gray"]))
