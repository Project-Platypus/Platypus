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

# A few implementation notes:
#
#   1. The input, a collection of solutions, can be a list, iterator,
#      generator, view, etc.
#
#   2. Do not modify the input!  Create a copy with `list(solutions)` if
#      required.
#
#   3. Use `functools.cmp_to_key()` to convert a comparison function to an
#      equivalent key function.
#
#   4. If no key is provided, the objective values are used.  This results
#      in lexicographical ordering when sorting.

def objectives_key(solution):
    return tuple(solution.objectives)

def variables_key(solution):
    return tuple([solution.problem.types[i].decode(solution.variables[i]) for i in range(solution.problem.nvars)])

def fitness_key(solution):
    return solution.fitness

def rank_key(solution):
    return solution.rank

def crowding_distance_key(solution):
    return solution.crowding_distance

def objective_value_at_index(index):
    return lambda solution: solution.objectives[index]

def feasible(solutions):
    return (x for x in solutions if x.feasible)

def infeasible(solutions):
    return (x for x in solutions if not x.feasible)

def _unique(solutions, key=objectives_key):
    seen = set()

    for solution in solutions:
        kval = key(solution)
        if kval not in seen:
            seen.add(kval)
            yield solution

def unique(solutions, key=objectives_key):
    """Returns the unique solutions.

    Parameters
    ----------
    solutions : iterable
        The list of solutions.
    key: callable
        Returns the key used to identify unique solutions.
    """
    return list(_unique(solutions, key))

def group(solutions, key=objectives_key):
    """Groups solutions by the given key.

    Returns a mapping from each unique key to a list of solutions with that
    key value.

    Parameters
    ----------
    solutions: iterable
        The collection of solutions being grouped.
    key: callable
        Returns the key used for grouping, where solutions with identical
        values are grouped together.
    """
    result = {}

    for solution in solutions:
        kval = key(solution)
        if kval in result:
            result[kval].append(solution)
        else:
            result[kval] = [solution]

    return result

def truncate(solutions, size, key=objectives_key, reverse=False):
    """Truncates the population down to the given size.

    Parameters
    ----------
    solutions: iterable
        The collection of solutions being truncated.
    size: int
        The number of solutions to return.
    key: callable
        Returns the key used for truncation.
    reverse: bool
        If True, reverse the ordering to truncate the smallest keys first.
    """
    return sorted(solutions, key=key, reverse=reverse)[:size]

def _prune(solutions, size, key=objectives_key, reverse=False):
    remaining = list(solutions)

    while size > 0 and len(remaining) > 0:
        # TODO: Optimize by replacing sorting with a linear scan
        remaining.sort(key=key, reverse=reverse)
        item = remaining.pop(0)
        size = size - 1
        yield item

def prune(solutions, size, key=objectives_key, reverse=False):
    """Prunes the population down to the given size.

    Similar to `truncate` except the key is re-evaluated after each solution
    is removed from the population.

    Parameters
    ----------
    solutions: iterable
        The collection of solutions being pruned.
    size: int
        The number of solutions to return.
    key: callable
        Returns the key used for pruning.
    reverse: bool
        If True, reverse the ordering to prune the smallest keys first.
    """
    return list(_prune(solutions, size, key, reverse))
