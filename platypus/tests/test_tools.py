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
from .._tools import remove_keys, only_keys, only_keys_for


class TestDictMethods(unittest.TestCase):

    def test_remove_keys(self):
        self.assertEqual({}, remove_keys({}))
        self.assertEqual({}, remove_keys({}, "a", "b"))
        self.assertEqual({}, remove_keys({"a": "remove"}, "a", "b"))
        self.assertEqual({"c": "keep"}, remove_keys({"a": "remove", "c": "keep"}, "a", "b"))
        self.assertEqual({"a": "keep"}, remove_keys({"a": "keep"}))

    def test_only_keys(self):
        self.assertEqual({}, only_keys({}))
        self.assertEqual({}, only_keys({}, "a", "b"))
        self.assertEqual({"a": "keep"}, only_keys({"a": "keep", "b": "remove"}, "a"))

    def _test_func_pos(self, a):
        pass

    def _test_func_def(self, a=5):
        pass

    def test_keys_for(self):
        self.assertEqual({}, only_keys_for({}, self._test_func_pos))
        self.assertEqual({"a": "keep"}, only_keys_for({"a": "keep", "b": "remove"}, self._test_func_pos))
        self.assertEqual({}, only_keys_for({}, self._test_func_def))
        self.assertEqual({"a": "keep"}, only_keys_for({"a": "keep", "b": "remove"}, self._test_func_def))
