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

import json
from .core import Algorithm, Archive, FixedLengthArray, Problem, Solution

def load_objectives(file, problem=None):
    """Loads objective values from a file.

    Parameters
    ----------
    file : str
        The file name.
    problem : Problem, optional
        The problem definition.  If :code:`None`, a placeholder is used.
    """
    result = []

    with open(file, "r") as f:
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
    file : str
        The file name.
    solutions : iterable of Solution
        The solutions to save.
    """
    with open(file, "w") as f:
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

def save_json(file, solutions, **kwargs):
    """Converts the solutions to JSON and saves to a file.

    Parameters
    ----------
    file : str
        The file name.
    solutions : object
        The solutions, archive, or algorithm.
    **kwargs : dict
        Additional arguments passed to the JSON library, such as formatting
        options.
    """
    with open(file, "w") as f:
        json.dump(solutions, f, cls=PlatypusJSONEncoder, **kwargs)

def load_json(file, problem=None, **kwargs):
    """Converts the solutions to JSON and saves to a file.

    Parameters
    ----------
    file : str
        The file name.
    problem : Problem, optional
        The problem definition.  If :code:`None`, a placeholder is used.
    **kwargs : dict
        Additional arguments passed to the JSON library.
    """
    with open(file, "r") as f:
        return json.load(f, cls=PlatypusJSONDecoder, problem=problem, **kwargs)
