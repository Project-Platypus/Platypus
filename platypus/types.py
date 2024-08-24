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

import copy
import math
import random
from abc import ABCMeta, abstractmethod

class Type(metaclass=ABCMeta):
    """The type of a decision variable.

    The type of a decision variable defines its bounds, provides a mechanism to
    produce a random value within those bounds, and defines any encoding /
    decoding to convert between the "value" and the internal representation.

    An example of the value differing from the internal representation
    is binary integers, where the value is an integer (e.g., 27) but its
    internal representation is a binary string (e.g., "11011" or in Python
    [True, True, False, True, True]).

    Subclasses should override __repr__ and __str__ to provide a human
    readable representation of the type.  The current standard is to
    return "TypeName(Arg1, Arg2, ...)".
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def rand(self):
        """Produces a random but valid encoded value for this type."""
        raise NotImplementedError()

    def encode(self, value):
        """Encodes a value into its internal representation."""
        return value

    def decode(self, value):
        """Decodes a value from its internal representation."""
        return value

class Real(Type):
    """Represents a floating-point value with min and max bounds.

    Attributes
    ----------
    min_value : int
        The minimum value (inclusive)
    max_value: int
        The maximum value (inclusive)
    """

    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def rand(self):
        return random.uniform(self.min_value, self.max_value)

    def __str__(self):
        return f"Real({self.min_value:f}, {self.max_value:f})"

class Binary(Type):
    """Represents a binary string.

    Binary strings are useful for problems dealing with subsets, where from a
    set of N items a subset of 0 to N elements are selected.  For example, see
    the Knapsack problem.

    Internally, in Python, the binary string is stored as a list of boolean
    values, where False represents the 0 (off) bit and and True represents the
    1 (on) bit.

    Attributes
    ----------
    nbits : int
        The number of bits.
    """

    def __init__(self, nbits):
        super().__init__()
        self.nbits = nbits

    def rand(self):
        return [random.choice([False, True]) for _ in range(self.nbits)]

    def __str__(self):
        return f"Binary({self.nbits})"

class Integer(Binary):
    """Represents an integer value with min and max bounds.

    Integers extend the Binary representation and encodes the integer as a
    gray-encoded binary value.  The gray-encoding ensures that adjacent
    integers (e.g., i and i+1) differ only by one bit.

    Given max_value and min_value, the underlying representation chooses the
    minimum number of bits required to store the integer in a binary string.
    If max_value-min_value is a power of 2, that each binary string maps to
    an integer value.  If max_value-min_value is not a power of 2, then some
    integers will have two binary strings mapping to the value, meaning those
    values have a slightly higher probability of occurrence.

    Attributes
    ----------
    min_value : int
        The minimum value (inclusive)
    max_value: int
        The maximum value (inclusive)
    nbits: int
        The number of bits used by the underlying representation.
    """

    def __init__(self, min_value, max_value):
        super().__init__(int(math.log(int(max_value)-int(min_value), 2)) + 1)
        self.min_value = int(min_value)
        self.max_value = int(max_value)

    def rand(self):
        return self.encode(random.randint(self.min_value, self.max_value))

    def encode(self, value):
        return bin2gray(int2bin(value-self.min_value, self.nbits))

    def decode(self, value):
        value = bin2int(gray2bin(value))

        if value > self.max_value-self.min_value:
            value -= self.max_value-self.min_value

        return self.min_value + value

    def __str__(self):
        return f"Integer({self.min_value}, {self.max_value})"

class Permutation(Type):
    """Represents a permutation.

    Permutations are stored as a list of elements in a specific order.  All
    elements will appear in the list exactly once.  For example, this is used
    to represent the traversal through a graph, such as for the Traveling
    Salesman Problem.

    Examples
    --------
        # A permutation of integers 0 through 9.
        Permutation(range(10))

        # A permutation of tuples.
        Permutation([(a1, a2), (b1, b2), (c1, c2), (d1, d2)])

    Attributes
    ----------
    elements : list of objects
        The list of elements that appear in the permutation.
    """

    def __init__(self, elements):
        super().__init__()
        self.elements = list(elements)

    def rand(self):
        elements = copy.deepcopy(self.elements)
        random.shuffle(elements)
        return elements

    def __str__(self):
        return f"Permutation({len(self.elements)})"

class Subset(Type):
    """Represents a fixed-size subset.

    Use a subset when you must select K elements from a collection of N items.
    Use a binary string when you can select any number of elements (0
    through N) from a collection of N items.

    Examples
    --------
        # Pick any two numbers between 0 and 9, without repeats.
        Subset(range(10), 2)

    Attributes
    ----------
    elements : list of objects
        The set of elements.
    size : int
        The size of the subset.
    """

    def __init__(self, elements, size):
        super().__init__()
        self.elements = list(elements)
        self.size = size

    def rand(self):
        indices = list(range(len(self.elements)))
        random.shuffle(indices)
        return [self.elements[i] for i in indices[:self.size]]

    def __str__(self):
        return f"Subset({len(self.elements)}, {self.size})"


# Modified from RosettaCode.org (http://rosettacode.org/wiki/Gray_code#Python)
def int2bin(n, nbits):
    """Converts an integer into a binary string.

    Parameters
    ----------
    n : int
        The integer value.
    nbits:
        The number of bits used to encode the value.
    """
    bits = []

    while n:
        n, remainder = divmod(n, 2)
        bits.insert(0, bool(remainder))

    while len(bits) < nbits:
        bits.insert(0, False)

    return bits

def bin2int(bits):
    """Converts a binary string into its integer value.

    Parameters
    ----------
    bits : list or tuple of bool
        The binary string as a list of True/False values.
    """
    i = 0

    for bit in bits:
        i = i * 2 + bit

    return i

def bin2gray(bits):
    """Converts a binary string into a gray-encoded binary string.

    Parameters
    ----------
    bits : list or tuple of bool
        The binary string as a list of True/False values.
    """
    return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

def gray2bin(bits):
    """Converts a gray-encoded binary string into a binary string.

    Parameters
    ----------
    bits : list or tuple of bool
        The gray-encoded binary string as a list of True/False values.
    """
    b = [bits[0]]
    for nextb in bits[1:]:
        b.append(b[-1] ^ nextb)
    return b
