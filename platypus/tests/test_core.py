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
import random
import pytest

from .._math import POSITIVE_INFINITY
from ..core import (Archive, Constraint, Direction, EpsilonBoxArchive,
                    ParetoDominance, Problem, nondominated_prune,
                    nondominated_sort, nondominated_split,
                    nondominated_truncate, normalize, truncate_fitness)
from ..errors import PlatypusError, PlatypusWarning
from ..types import Real
from ._utils import createSolution

s0 = createSolution(0.0, 0.0)
s1 = createSolution(0.0, 1.0)
s2 = createSolution(0.5, 0.5)
s3 = createSolution(1.0, 0.0)
s4 = createSolution(0.75, 0.75)
s5 = createSolution(1.0, 1.0)

@pytest.fixture
def sample_population():
    population = [s1, s2, s3, s4, s5]
    random.shuffle(population)
    return population

def test_solution_deepcopy():
    orig = createSolution(4, 5)
    orig.constraint_violation = 2
    orig.evaluated = True

    clone = copy.deepcopy(orig)

    assert orig.problem == clone.problem
    assert 4 == clone.objectives[0]
    assert 5 == clone.objectives[1]
    assert 2 == clone.constraint_violation
    assert clone.evaluated

def test_solution_assignment_warning():
    s = createSolution(4, 5)
    with pytest.warns(PlatypusWarning):
        s.objectives = [1, 2]

def test_direction():
    assert Direction.MINIMIZE == Direction.to_direction(-1)
    assert Direction.MINIMIZE == Direction.to_direction("minimize")
    assert Direction.MINIMIZE == Direction.to_direction(Direction.MINIMIZE)
    assert Direction.MINIMIZE == Direction.to_direction(Problem.MINIMIZE)

    assert Direction.MAXIMIZE == Direction.to_direction(1)
    assert Direction.MAXIMIZE == Direction.to_direction("maximize")
    assert Direction.MAXIMIZE == Direction.to_direction(Direction.MAXIMIZE)
    assert Direction.MAXIMIZE == Direction.to_direction(Problem.MAXIMIZE)

def test_constraint_eq():
    constraint = Constraint("==0")
    assert 0.0 == constraint(0.0)
    assert 0.0 != constraint(1.0)
    assert 0.0 != constraint(-1.0)

    constraint = Constraint("==5")
    assert 0.0 == constraint(5.0)
    assert 0.0 != constraint(-5.0)
    assert 0.0 != constraint(10.0)

def test_constraint_lte():
    constraint = Constraint("<=0")
    assert 0.0 == constraint(0.0)
    assert 0.0 == constraint(-1.0)
    assert 0.0 != constraint(1.0)

def test_constraint_gte():
    constraint = Constraint(">=0")
    assert 0.0 == constraint(0.0)
    assert 0.0 == constraint(1.0)
    assert 0.0 != constraint(-1.0)

def test_constraint_lt():
    constraint = Constraint("<0")
    assert 0.0 != constraint(0.0)
    assert 0.0 != constraint(1.0)
    assert 0.0 == constraint(-1.0)

def test_constraint_gt():
    constraint = Constraint(">0")
    assert 0.0 != constraint(0.0)
    assert 0.0 == constraint(1.0)
    assert 0.0 != constraint(-1.0)

def test_constraint_neq():
    constraint = Constraint("!=0")
    assert 0.0 != constraint(0.0)
    assert 0.0 == constraint(1.0)
    assert 0.0 == constraint(-1.0)

def test_constraint_bad_operator():
    with pytest.raises(PlatypusError):
        Constraint("=!0")

def test_constraint_missing_operator():
    with pytest.raises(PlatypusError):
        Constraint("0")

def test_constraint_missing_value():
    with pytest.raises(PlatypusError):
        Constraint("<=")

def test_constraint_empty_string():
    with pytest.raises(PlatypusError):
        Constraint("")

def test_problem_single_assignment():
    problem = Problem(2, 2, 2)
    problem.types[:] = Real(0, 1)
    problem.directions[:] = Direction.MINIMIZE
    problem.constraints[:] = Constraint.LESS_THAN_ZERO

    assert len(problem.types) == 2
    assert len(problem.directions) == 2
    assert len(problem.constraints) == 2
    assert all([t is not None for t in problem.types])
    assert all([d is not None for d in problem.directions])
    assert all([c is not None for c in problem.constraints])

def test_problem_array_assignment():
    problem = Problem(2, 2, 2)
    problem.types[:] = [Real(0, 1), Real(0, 1)]
    problem.directions[:] = [Direction.MINIMIZE, Direction.MAXIMIZE]
    problem.constraints[:] = [Constraint.LESS_THAN_ZERO, Constraint.GREATER_THAN_ZERO]

    assert len(problem.types) == 2
    assert len(problem.directions) == 2
    assert len(problem.constraints) == 2
    assert all([t is not None for t in problem.types])
    assert all([d is not None for d in problem.directions])
    assert all([c is not None for c in problem.constraints])

def test_pareto_dominance():
    dominance = ParetoDominance()
    assert -1 == dominance.compare(s0, s1)
    assert 1 == dominance.compare(s1, s0)
    assert -1 == dominance.compare(s0, s3)
    assert 1 == dominance.compare(s3, s0)

def test_pareto_nondominance():
    dominance = ParetoDominance()
    assert 0 == dominance.compare(s1, s2)
    assert 0 == dominance.compare(s2, s1)
    assert 0 == dominance.compare(s2, s3)
    assert 0 == dominance.compare(s3, s2)
    assert 0 == dominance.compare(s1, s3)
    assert 0 == dominance.compare(s3, s1)

def test_archive_dominance():
    archive = Archive(ParetoDominance())
    archive += [s5, s0, s1]
    assert 1 == len(archive)
    assert s0 == archive[0]

def test_archive_nondominance():
    archive = Archive(ParetoDominance())
    archive += [s1, s2, s3]
    assert 3 == len(archive)

def test_ranking(sample_population):
    nondominated_sort(sample_population)
    assert 0 == s1.rank
    assert 0 == s2.rank
    assert 0 == s3.rank
    assert 1 == s4.rank
    assert 2 == s5.rank

def test_crowding(sample_population):
    nondominated_sort(sample_population)
    assert POSITIVE_INFINITY == s1.crowding_distance
    assert 2 == s2.crowding_distance
    assert POSITIVE_INFINITY == s3.crowding_distance
    assert POSITIVE_INFINITY == s4.crowding_distance
    assert POSITIVE_INFINITY == s5.crowding_distance

def test_split2(sample_population):
    nondominated_sort(sample_population)
    (first, second) = nondominated_split(sample_population, 2)

    assert 0 == len(first)
    assert 3 == len(second)
    assert s1 in second
    assert s2 in second
    assert s3 in second

def test_split3(sample_population):
    nondominated_sort(sample_population)
    (first, second) = nondominated_split(sample_population, 3)
    assert 3 == len(first)
    assert 0 == len(second)
    assert s1 in first
    assert s2 in first
    assert s3 in first

def test_split4(sample_population):
    nondominated_sort(sample_population)
    (first, second) = nondominated_split(sample_population, 4)
    assert 4 == len(first)
    assert 0 == len(second)
    assert s1 in first
    assert s2 in first
    assert s3 in first
    assert s4 in first

def test_truncate2(sample_population):
    nondominated_sort(sample_population)
    result = nondominated_truncate(sample_population, 2)
    assert 2 == len(result)
    assert s1 in result
    assert s3 in result

def test_truncate4(sample_population):
    nondominated_sort(sample_population)
    result = nondominated_truncate(sample_population, 4)
    assert 4 == len(result)
    assert s1 in result
    assert s2 in result
    assert s3 in result
    assert s4 in result

def test_prune2(sample_population):
    nondominated_sort(sample_population)
    result = nondominated_prune(sample_population, 2)
    assert 2 == len(result)
    assert s1 in result
    assert s3 in result

def test_prune4(sample_population):
    nondominated_sort(sample_population)
    result = nondominated_prune(sample_population, 4)
    assert 4 == len(result)
    assert s1 in result
    assert s2 in result
    assert s3 in result
    assert s4 in result

def test_truncate_fitness_max(sample_population):
    s1.fitness = 1
    s2.fitness = 5
    s3.fitness = 3
    s4.fitness = 4
    s5.fitness = 2
    result = truncate_fitness(sample_population, 3)
    assert 3 == len(result)
    assert s2 in result
    assert s3 in result
    assert s4 in result

def test_truncate_fitness_min(sample_population):
    s1.fitness = 1
    s2.fitness = 5
    s3.fitness = 3
    s4.fitness = 4
    s5.fitness = 2
    result = truncate_fitness(sample_population, 3, larger_preferred=False)
    assert 3 == len(result)
    assert s1 in result
    assert s3 in result
    assert s5 in result

def test_normalize():
    s1 = createSolution(0, 2)
    s2 = createSolution(2, 3)
    s3 = createSolution(1, 1)
    solutions = [s1, s2, s3]

    normalize(solutions)

    assert [0.0, 0.5] == s1.normalized_objectives
    assert [1.0, 1.0] == s2.normalized_objectives
    assert [0.5, 0.0] == s3.normalized_objectives

def test_improvements():
    s1 = createSolution(0.25, 0.25)     # Improvement 1 - First solution always counted as improvement
    s2 = createSolution(0.10, 0.10)     # Improvement 2 - Dominates prior solution and in new epsilon-box
    s3 = createSolution(0.24, 0.24)
    s4 = createSolution(0.09, 0.50)     # Improvement 3 - Non-dominated to all existing solutions
    s5 = createSolution(0.50, 0.50)
    s6 = createSolution(0.05, 0.05)     # Improvement 4 - Dominates prior solution and in new epsilon-box
    s7 = createSolution(0.04, 0.04)
    s8 = createSolution(0.02, 0.02)
    s9 = createSolution(0.00, 0.00)
    s10 = createSolution(-0.01, -0.01)  # Improvement 5 - Dominates prior solution and in new epsilon-box

    solutions = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    expectedImprovements = [1, 2, 2, 3, 3, 4, 4, 4, 4, 5]

    archive = EpsilonBoxArchive([0.1])

    for (s, i) in zip(solutions, expectedImprovements):
        archive.add(s)
        assert i == archive.improvements
