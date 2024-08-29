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

import os
import json
from .core import Algorithm, Archive, FixedLengthArray, Problem, Solution

def load_objectives(file, problem=None):
    """Loads objective values from a file.

    Parameters
    ----------
    file : str, bytes, or :class:`os.PathLike`
        The file.
    problem : Problem, optional
        The problem definition.  If :code:`None`, a placeholder is used.
    """
    result = []

    with open(os.fspath(file), "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            values = list(map(float, line.split()))

            if problem is None:
                problem = Problem(0, len(values))

            solution = Solution(problem)
            solution.objectives[:] = values
            result.append(solution)

    return result

def save_objectives(file, solutions):
    """Saves objective values to a file.

    Parameters
    ----------
    file : str, bytes, or :class:`os.PathLike`
        The file.
    solutions : iterable of Solution
        The solutions to save.
    """
    with open(os.fspath(file), "w") as f:
        for solution in solutions:
            f.write(" ".join(map(str, solution.objectives)))
            f.write("\n")

class PlatypusJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (Archive, FixedLengthArray)):
            return list(obj)
        if isinstance(obj, Algorithm):
            return {"algorithm": {"name": type(obj).__name__,
                                  "nfe": obj.nfe},
                    "problem": {"name": type(obj.problem).__name__,
                                "nvars": obj.problem.nvars,
                                "nobjs": obj.problem.nobjs,
                                "nconstrs": obj.problem.nconstrs,
                                "function": obj.problem.function,
                                "directions": obj.problem.directions,
                                "constraints": obj.problem.constraints},
                    "result": obj.result}
        if isinstance(obj, Solution):
            return {"variables": obj.variables,
                    "objectives": obj.objectives,
                    "constraints": obj.constraints}
        return super().default(obj)

class PlatypusJSONDecoder(json.JSONDecoder):

    def __init__(self, problem=None, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
        self.problem = problem

    def object_hook(self, d):
        if "problem" in d and "result" in d:
            if self.problem is None:
                self.problem = Problem(int(d["problem"]["nvars"]),
                                       int(d["problem"]["nobjs"]),
                                       int(d["problem"]["nconstrs"]))
                self.problem.directions[:] = d["problem"]["directions"]
                self.problem.constraints[:] = d["problem"]["constraints"]

            return d["result"]

        if "variables" in d and "objectives" in d and "constraints" in d:
            if self.problem is None:
                self.problem = Problem(len(d["variables"]),
                                       len(d["objectives"]),
                                       len(d["constraints"]))

            solution = Solution(self.problem)
            solution.variables[:] = d["variables"]
            solution.objectives[:] = d["objectives"]
            solution.constraints[:] = d["constraints"]

            solution.constraint_violation = sum([abs(f(x)) for (f, x) in zip(solution.problem.constraints, solution.constraints)])
            solution.feasible = solution.constraint_violation == 0.0
            solution.evaluated = True

            return solution

        return d

def dump(obj, fp, indent=None):
    """Dumps the object to JSON.

    This overrides :meth:`json.dump` to convert Platypus types to JSON.  Use
    the corresponding :meth:`load` method in this module to load the JSON back
    into memory.

    Parameters
    ----------
    obj : object
        The object to dump to JSON.
    fp : file-like
        A file-like object, typically created by :meth:`open`.
    indent:
        Controls the formatting of the JSON fle, see :meth:`json.dump`.
    """
    json.dump(obj, fp, cls=PlatypusJSONEncoder, indent=indent)

def load(fp, problem=None):
    """Loads the JSON data.

    Parameters
    ----------
    fp : file-like
        A file-like object, typically created by :meth:`open`.
    problem : Problem, optional
        Optional problem definition.  If not set, a placeholder is used.
    """
    return json.load(fp, cls=PlatypusJSONDecoder, problem=problem)

def save_json(file, solutions, indent=None):
    """Converts the solutions to JSON and saves to a file.

    If given an :class:`Algorithm` object, extra information about the
    algorithm state and problem are stored in the JSON; however, this
    is for informational purposes only and can not be read back.

    Parameters
    ----------
    file : str, bytes, or :class:`os.PathLike`
        The file.
    solutions : object
        The solutions, archive, or algorithm.
    indent:
        Controls the formatting of the JSON fle, see :meth:`json.dump`.
    """
    with open(os.fspath(file), "w") as f:
        dump(solutions, f, indent=indent)

def load_json(file, problem=None):
    """Loads the solutions stored in a JSON file.

    Parameters
    ----------
    file : str, bytes, or :class:`os.PathLike`
        The file.
    problem : Problem, optional
        The problem definition.  If :code:`None`, a placeholder is used.
    """
    with open(os.fspath(file), "r") as f:
        return load(f, problem=problem)
