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

from .core import Solution

def load_objectives(file, problem):
    """Loads objective values from a file.
    
    Parameters
    ----------
    file : str
        The file name.
    problem : Problem
        The problem definition.
    """
    result = []

    with open(file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            solution = Solution(problem)
            solution.objectives[:] = list(map(float, line.split()))
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
