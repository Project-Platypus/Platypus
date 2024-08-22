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

import inspect


def remove_keys(d, *keys):
    """Returns a new dictionary with the given keys removed.

    Parameters
    ----------
    d : dict
        The original dictionary.
    keys : list of keys
        The keys to remove.  If the key is not found in the dictionary, it is
        ignored.
    """
    result = dict(d)
    for key in keys:
        result.pop(key, None)
    return result

def only_keys(d, *keys):
    """Returns a new dictionary containing only the given keys.

    Parameters
    ----------
    d : dict
        The original dictionary.
    keys: list of keys
        The keys to keep.  If a key is not found in the dictionary, it is
        ignored.
    """
    result = dict()
    for key in keys:
        if key in d:
            result[key] = d[key]
    return result

def only_keys_for(d, func):
    """Returns a new dictionary containing only keys matching function arguments.

    Parameters
    ----------
    d : dict
        The original dictionary.
    func: callable
        The function.
    """
    return only_keys(d, *inspect.getfullargspec(func)[0])
