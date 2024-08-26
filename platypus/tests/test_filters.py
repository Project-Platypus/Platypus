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
from .test_core import createSolution
from ..filters import unique, group, truncate, matches

def iterator(*args):
    return iter(args)

def generator(*args):
    return (x for x in args)

def view(*args):
    return {x: x for x in args}.keys()

class FilterTestCase(unittest.TestCase, metaclass=ABCMeta):

    s1 = createSolution(0.0, 1.0)
    s2 = createSolution(1.0, 0.0)
    s3 = createSolution(0.0, 1.0)

    @abstractmethod
    def filter(self, solutions):
        raise NotImplementedError()

    def test_list_empty(self):
        self.assertEqual(self.empty_result, self.filter([]))

    def test_list_single_item(self):
        self.assertEqual(self.single_item_result, self.filter([self.s1]))

    def test_list_multiple_items(self):
        self.assertEqual(self.multiple_item_result, self.filter([self.s1, self.s2, self.s3]))

    def test_iterator_empty(self):
        self.assertEqual(self.empty_result, self.filter(iterator()))

    def test_iterator_single_item(self):
        self.assertEqual(self.single_item_result, self.filter(iterator(self.s1)))

    def test_iterator_multiple_items(self):
        self.assertEqual(self.multiple_item_result, self.filter(iterator(self.s1, self.s2, self.s3)))

    def test_generator_empty(self):
        self.assertEqual(self.empty_result, self.filter(generator()))

    def test_generator_single_item(self):
        self.assertEqual(self.single_item_result, self.filter(generator(self.s1)))

    def test_generator_multiple_items(self):
        self.assertEqual(self.multiple_item_result, self.filter(generator(self.s1, self.s2, self.s3)))

    def test_view_empty(self):
        self.assertEqual(self.empty_result, self.filter(view()))

    def test_view_single_item(self):
        self.assertEqual(self.single_item_result, self.filter(view(self.s1)))

    def test_view_multiple_items(self):
        self.assertEqual(self.multiple_item_result, self.filter(view(self.s1, self.s2, self.s3)))

class TestUnique(FilterTestCase):

    def setUp(self):
        self.empty_result = []
        self.single_item_result = [self.s1]
        self.multiple_item_result = [self.s1, self.s2]

    def filter(self, solutions):
        return unique(solutions)

class TestGroup(FilterTestCase):

    def setUp(self):
        self.empty_result = {}
        self.single_item_result = {(0.0, 1.0): [self.s1]}
        self.multiple_item_result = {(0.0, 1.0): [self.s1, self.s3], (1.0, 0.0): [self.s2]}

    def filter(self, solutions):
        return group(solutions)

class TestTruncate(FilterTestCase):

    def setUp(self):
        self.empty_result = []
        self.single_item_result = [self.s1]
        self.multiple_item_result = [self.s2]

    def filter(self, solutions):
        return truncate(solutions, 1, key=lambda x: x.objectives[1])

class TestMatches(FilterTestCase):

    def setUp(self):
        self.empty_result = []
        self.single_item_result = [self.s1]
        self.multiple_item_result = [self.s1, self.s3]

    def filter(self, solutions):
        return matches(solutions, 1.0, key=lambda x: x.objectives[1])
