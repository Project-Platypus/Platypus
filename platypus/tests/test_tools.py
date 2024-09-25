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

from .._tools import (coalesce, only_keys, only_keys_for, parse_cli_keyvalue,
                      remove_keys, type_cast)


def mock_func_pos(a):
    pass

def mock_func_def(a=5):
    pass

def mock_func_kwonly(*, a=5):
    pass

def test_remove_keys():
    assert {} == remove_keys({})
    assert {} == remove_keys({}, "a", "b")
    assert {} == remove_keys({"a": "remove"}, "a", "b")
    assert {"c": "keep"} == remove_keys({"a": "remove", "c": "keep"}, "a", "b")
    assert {"a": "keep"} == remove_keys({"a": "keep"})

def test_only_keys():
    assert {} == only_keys({})
    assert {} == only_keys({}, "a", "b")
    assert {"a": "keep"} == only_keys({"a": "keep", "b": "remove"}, "a")

def test_keys_for():
    assert {} == only_keys_for({}, mock_func_pos)
    assert {"a": "keep"} == only_keys_for({"a": "keep", "b": "remove"}, mock_func_pos)
    assert {} == only_keys_for({}, mock_func_def)
    assert {"a": "keep"} == only_keys_for({"a": "keep", "b": "remove"}, mock_func_def)
    assert {} == only_keys_for({}, mock_func_kwonly)
    assert {"a": "keep"} == only_keys_for({"a": "keep", "b": "remove"}, mock_func_kwonly)

def test_parse_cli_keyvalue():
    assert {} == parse_cli_keyvalue([])
    assert {"a": "2"} == parse_cli_keyvalue(["a=2"])
    assert {"a": "2", "b": "foo"} == parse_cli_keyvalue(["a=2", "b=foo"])

def test_type_cast():
    args = {"a": "2", "b": "foo"}
    assert args == type_cast(args, mock_func_pos)
    assert {"a": 2, "b": "foo"} == type_cast(args, mock_func_def)
    assert {"a": 2, "b": "foo"} == type_cast(args, mock_func_kwonly)

def test_coalesce():
    assert coalesce() is None
    assert coalesce(None, None) is None
    assert "foo" == coalesce(None, "foo", "bar", None)

    with pytest.raises(ValueError):
        coalesce(None, None, throw_if_none=True)
