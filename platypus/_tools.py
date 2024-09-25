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
import logging
import types

from .errors import PlatypusError

LOGGER = logging.getLogger("Platypus")

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
    argspec = inspect.getfullargspec(func)
    args = argspec.args + argspec.kwonlyargs
    return only_keys(d, *args)

def log_args(args, target):
    """Logs the arguments."""
    for k, v in args.items():
        LOGGER.info("Setting %s=%s on %s", k, v, target)

def parse_cli_keyvalue(args):
    """Parses CLI key=value pairs into a dictionary.

    Parameters
    ----------
    args : list of str
        List of CLI key=value pair arguments

    Returns
    -------
    The arguments parsed into a dictionary with :code:`d[key]=value`.
    """
    d = {}
    for arg in args:
        try:
            k, v = arg.split("=")
            d[k] = v
        except ValueError:
            raise PlatypusError(f"expected key=value pair, given {arg}")
    return d

def _type_cast_value(name, value, argspec):
    if name in argspec.annotations:
        return _type_cast_type(value, argspec.annotations[name])
    if name in argspec.args and argspec.defaults:
        return _type_cast_type(value, type(argspec.defaults[argspec.args.index(name) - (len(argspec.args) - len(argspec.defaults))]))
    if name in argspec.kwonlyargs and argspec.kwonlydefaults:
        return _type_cast_type(value, type(argspec.kwonlydefaults[name]))
    return value

def _type_cast_type(value, argtype):
    if value is None:
        return None
    if hasattr(argtype, "__call__"):
        return argtype(value)
    if argtype == types.UnionType:
        for t in argtype.__args__:
            return _type_cast_type(value, t)
    return value

def type_cast(d, func):
    """Attempts to convert the values in a dictionary to the appropriate types.

    Parameters
    ----------
    d : dict
        The dictionary.
    func: callable
        The function.

    Returns
    -------
    A new dictionary with the values cast to the func argument types.  If the
    conversion could not be determined, the original value is returned.
    """
    argspec = inspect.getfullargspec(func)
    result = {}

    for k, v in d.items():
        result[k] = _type_cast_value(k, v, argspec)

    return result

def coalesce(*args, throw_if_none=False):
    """Returns the first value, skipping any that are None.

    Parameters
    ----------
    args
        The values, of which at least one should be set.
    throw_if_none : bool
        If :code:`true`, throws if no value is set.
    """
    for a in args:
        if a is not None:
            return a
    if throw_if_none:
        raise ValueError("expected at least one value")
    return None
