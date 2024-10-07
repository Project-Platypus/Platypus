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
import functools
import inspect
import itertools
import math
import operator
import re
import time
import warnings
from abc import ABCMeta, abstractmethod
from enum import Enum

from ._math import EPSILON, POSITIVE_INFINITY
from .config import PlatypusConfig
from .errors import PlatypusError, PlatypusWarning
from .evaluator import Job
from .filters import (crowding_distance_key, fitness_key, matches,
                      objective_value_at_index, rank_key, truncate, unique)


class FixedLengthArray:

    def __init__(self, size, default_value=None, convert=None):
        super().__init__()
        self._size = size
        if convert is not None:
            self._data = [convert(default_value) for _ in range(size)]
        else:
            self._data = [default_value]*size
        self.convert = convert

    def __len__(self):
        return self._size

    def __setitem__(self, index, value):
        if self.convert is not None:
            value = self.convert(value)

        if isinstance(index, slice):
            indices = range(*index.indices(self._size))

            if hasattr(value, "__len__") and len(value) == len(indices):
                for i, entry in enumerate(indices):
                    self._data[entry] = value[i]
            else:
                for entry in range(*index.indices(self._size)):
                    self._data[entry] = value
        else:
            self._data[index] = value

    def __getitem__(self, index):
        return self._data[index]

    def __str__(self):
        return "[" + ", ".join(list(map(str, self._data))) + "]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, FixedLengthArray):
            return self._size == other._size and self._data == other._data
        return NotImplemented

class WarnOnOverwriteMixin:
    """Mixin for classes using FixedLengthArray to warn on direct assignment."""

    def __setattr__(self, name, value):
        if hasattr(self, name) and isinstance(getattr(self, name), FixedLengthArray) and not isinstance(value, FixedLengthArray):
            warnings.warn(f"Avoid assigning attribute '{name}' directly, use indices '{name}[:]=...' instead",
                          category=PlatypusWarning, stacklevel=2)
        super().__setattr__(name, value)

class Direction(Enum):
    """Defines the optimization direction for an objective."""

    MINIMIZE = -1
    MAXIMIZE = 1

    @classmethod
    def to_direction(cls, obj):
        """Converts the given object to an optimization direction.

        Parameters
        ----------
        obj : str, int, or Direction
            The object representing a direction.
        """
        if isinstance(obj, str):
            try:
                return Direction[obj.upper()]
            except KeyError:
                raise PlatypusError(f"{obj} is not a valid direction, valid values: {', '.join([d.name for d in Direction])}")
        elif isinstance(obj, (list, tuple)):
            return [Direction.to_direction(d) for d in obj]
        else:
            return Direction(obj)

class Problem(WarnOnOverwriteMixin):
    """Class representing a problem.

    Attributes
    ----------
    nvars: int
        The number of decision variables
    nobjs: int
        The number of objectives.
    nconstrs: int
        The number of constraints.
    function: callable
        The function used to evaluate the problem.  If no function is given,
        it is expected that the evaluate method is overridden.
    types: FixedLengthArray of Type
        The type of each decision variable.  The type describes the bounds and
        encoding/decoding required for each decision variable.
    directions: FixedLengthArray of Direction
        The optimization direction of each objective, either MINIMIZE (-1) or
        MAXIMIZE (1)
    constraints: FixedLengthArray of Constraint
        Describes the types of constraints as an equality or inequality.  The
        default requires each constraint value to be 0.
    """

    # These values are being deprecated, use Direction instead
    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self, nvars, nobjs, nconstrs=0, function=None):
        """Create a new problem.

        Problems can be constructed by either subclassing and overriding the
        evaluate method or passing in a function of the form::

            def func_name(vars):
                # vars is a list of the decision variable values
                return (objs, constrs)

        Parameters
        ----------
        nvars: int
            The number of decision variables.
        nobjs: int
            The number of objectives.
        nconstrs: int (default 0)
            The number of constraints.
        function: callable (default None)
            The function that is used to evaluate the problem.
        """
        super().__init__()
        self.nvars = nvars
        self.nobjs = nobjs
        self.nconstrs = nconstrs
        self.function = function
        self.types = FixedLengthArray(nvars)
        self.directions = FixedLengthArray(nobjs, Direction.MINIMIZE, Direction.to_direction)
        self.constraints = FixedLengthArray(nconstrs, "==0", Constraint.to_constraint)

    def __call__(self, solution):
        """Evaluate the solution.

        This method is responsible for decoding the decision variables,
        invoking the evaluate method, and updating the solution.

        Parameters
        ----------
        solution: Solution
            The solution to evaluate.
        """
        problem = solution.problem
        solution.variables[:] = [problem.types[i].decode(solution.variables[i]) for i in range(problem.nvars)]

        self.evaluate(solution)

        solution.variables[:] = [problem.types[i].encode(solution.variables[i]) for i in range(problem.nvars)]
        solution.constraint_violation = sum([abs(f(x)) for (f, x) in zip(solution.problem.constraints, solution.constraints)])
        solution.feasible = solution.constraint_violation == 0.0
        solution.evaluated = True

    def evaluate(self, solution):
        """Evaluates the problem.

        By default, this method calls the function passed to the constructor.
        Alternatively, a problem can subclass and override this method.  When
        overriding, this method is responsible for updating the objectives
        and constraints stored in the solution.

        Parameters
        ----------
        solution: Solution
            The solution to evaluate.
        """
        if self.function is None:
            raise PlatypusError("function not defined")

        if self.nconstrs > 0:
            (objs, constrs) = self.function(solution.variables)
        else:
            objs = self.function(solution.variables)
            constrs = []

        if not hasattr(objs, "__getitem__"):
            objs = [objs]

        if not hasattr(constrs, "__getitem__"):
            constrs = [constrs]

        if len(objs) != self.nobjs:
            raise PlatypusError("incorrect number of objectives: expected %d, received %d" % (self.nobjs, len(objs)))

        if len(constrs) != self.nconstrs:
            raise PlatypusError("incorrect number of constraints: expected %d, received %d" % (self.nconstrs, len(constrs)))

        solution.objectives[:] = objs
        solution.constraints[:] = constrs

class Generator(metaclass=ABCMeta):
    """Abstract class for generating initial populations."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, problem):
        """Generates a single, random solution for the problem.

        Parameters
        ----------
        problem : Problem
            The problem being optimized.
        """
        raise NotImplementedError()

class Variator(metaclass=ABCMeta):
    """Abstract class for variation operators (crossover and mutation).

    Variation operators must not modify the parents!  Instead, create a copy
    of the parents, set :code:`evaluated` to :code:`False`, and make any
    modifications on the copy::

        child = copy.deepcopy(parent)
        child.evaluated = False

    Parameters
    ----------
    arity : int
        The operator arity, or number of required parents.
    """

    def __init__(self, arity):
        super().__init__()
        self.arity = arity

    @abstractmethod
    def evolve(self, parents):
        """Evolves the parents to produce offspring.

        Parameters
        ----------
        parents : list of Solution
            The parent solutions.

        Returns
        -------
        The offspring.
        """
        raise NotImplementedError()

class Mutation(Variator, metaclass=ABCMeta):
    """Abstract class for mutation operators.

    Mutation is just a special :class:`Variator` with an arity of 1.  While
    not required, mutation operators typically also produce a single offspring.
    """

    def __init__(self):
        super().__init__(1)

    def evolve(self, parents):
        if hasattr(parents, "__iter__"):
            return list(map(self.mutate, parents))
        else:
            return self.mutate(parents)

    @abstractmethod
    def mutate(self, parent):
        """Mutates the given parent.

        Parameters
        ----------
        parent : Solution
            The parent to mutate.

        Returns
        -------
        The offspring.
        """
        raise NotImplementedError()

class Selector(metaclass=ABCMeta):
    """Abstract class for selection operators."""

    def __init__(self):
        super().__init__()

    def select(self, n, population):
        """Selects N members from the population.

        This default implementation operates "with replacement", meaning
        solutions can be selected multiple times.  Subclasses are allowed to
        modify this behavior.

        Parameters
        ----------
        n : int
            The number of solutions to select from the population.
        population: list of Solution
            The population of solutions.
        """
        return list(map(self.select_one, itertools.repeat(population, n)))

    @abstractmethod
    def select_one(self, population):
        """Selects a single member from the population.

        Parameters
        ----------
        population: list of Solution
            The population of solutions.
        """
        raise NotImplementedError()

class TerminationCondition(metaclass=ABCMeta):
    """Abstract class for defining termination conditions."""

    def __init__(self):
        super().__init__()

    def __call__(self, algorithm):
        return self.shouldTerminate(algorithm)

    def initialize(self, algorithm):
        """Initializes this termination condition.

        This method is called once at the start of a run, and should be used
        to record any initial state.  Also consider that users can call
        :code:`run` multiple times, where each invocation continues the
        execution of the run from where it left off.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm instance.
        """
        pass

    @abstractmethod
    def shouldTerminate(self, algorithm):
        """Checks if the algorithm should terminate.

        This method is called after each iteration.  The defintion of one
        "iteration" in specific to each algorithm.  For example, a
        generational algorithm typically produces and evaluates multiple
        offspring each iteration.  Consequently, a run might not stop exactly
        where the termination condition indicates.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm instance.

        Returns
        -------
        :code:`True` if the termination condition is satisfied and the run
        should stop; :code:`False` otherwise.
        """
        return False

class MaxEvaluations(TerminationCondition):
    """Termination condition based on the maximum number of function
    evaluations.

    Parameters
    ----------
    nfe : int
        The maximum number of function evaluations to run the algorithm.
    """
    def __init__(self, nfe):
        super().__init__()
        self.nfe = nfe
        self.starting_nfe = 0

    def initialize(self, algorithm):
        self.starting_nfe = algorithm.nfe

    def shouldTerminate(self, algorithm):
        return algorithm.nfe - self.starting_nfe >= self.nfe

class MaxTime(TerminationCondition):
    """Termination condition based on the maximum elapsed time.

    Parameters
    ----------
    max_time : float
        The duration, in seconds, to run the algorithm.
    """

    def __init__(self, max_time):
        super().__init__()
        self.max_time = max_time
        self.start_time = time.time()

    def initialize(self, algorithm):
        self.start_time = time.time()

    def shouldTerminate(self, algorithm):
        return time.time() - self.start_time >= self.max_time

def _constraint_eq(x, y):
    return abs(x - y)

def _constraint_leq(x, y):
    return 0 if x <= y else abs(x - y)

def _constraint_geq(x, y):
    return 0 if x >= y else abs(x - y)

def _constraint_neq(x, y):
    return 0 if x != y else 1

def _constraint_lt(x, y, delta=0.0001):
    return 0 if x < y else abs(x - y) + delta

def _constraint_gt(x, y, delta=0.0001):
    return 0 if x > y else abs(x - y) + delta

class Constraint:
    """Defines an constraint on an optimization problem.

    A constraint can be defined in several ways.  First, with a given operator
    and value::

        Constraint("<=", 10)

    Second, by providing a string with the operator and value together::

        Constraint("<= 10")

    Third, by providing a function to compute the constraint value, where any
    non-zero value is considered a constraint violation::

        Constraint(lambda x : 0 if x <= 10 else math.abs(10 - x))

    Parameters
    ----------
    op : str or Callable
        The operator, such as :code:`"<="`, the full constraint including the
        value, such as :code:`"<=0"`, or a function to compute the constraint
        value.
    value : float, optional
        The value of the constraint.  This is only required when passing an
        operator without a value to :code:`op`.
    """

    OPERATORS = {
        "==": _constraint_eq,
        "<=": _constraint_leq,
        ">=": _constraint_geq,
        "!=": _constraint_neq,
        "<": _constraint_lt,
        ">": _constraint_gt
    }

    EQUALS_ZERO = "==0"
    LEQ_ZERO = "<=0"
    GEQ_ZERO = ">=0"
    LESS_THAN_ZERO = "<0"
    GREATER_THAN_ZERO = ">0"

    def __init__(self, op, value=None):
        super().__init__()

        if value is not None:
            # Passing value as a second argument
            self.op = op + str(value)
            self.function = functools.partial(Constraint.OPERATORS[op], y=float(value))
        elif isinstance(op, Constraint):
            # Passing a constraint object
            self.op = op.op
            self.function = op.function
        elif hasattr(op, "__call__"):
            # Passing a function that returns 0 if feasible and non-zero if not feasible
            self.op = op
            self.function = op
        else:
            self.op = op
            match = re.match(r"^([<>=!]+)\s*([^\s<>=!]+)$", op)

            try:
                if match:
                    self.function = functools.partial(Constraint.OPERATORS[match.group(1)], y=float(match.group(2)))
                else:
                    raise PlatypusError(f"{op} is not a valid constraint, unable to parse expression")
            except Exception as e:
                raise PlatypusError(f"{op} is not a valid constraint", e)

    def __call__(self, value):
        return self.function(value)

    @classmethod
    def to_constraint(cls, obj):
        """Converts the given object to a constraint.

        Parameters
        ----------
        obj : str, Constraint, or a list of constraints
            The constraint given as a string, Constraint, or a list / tuple
            of objects representing constraints.
        """
        if isinstance(obj, Constraint):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [Constraint.to_constraint(c) for c in obj]
        else:
            return Constraint(obj)

class Solution(WarnOnOverwriteMixin):
    """Class representing a solution to a problem.

    Parameters
    ----------
    problem: Problem
        The problem.

    Attributes
    ----------
    variables: FixedLengthArray of objects
        The values of the variables.
    objectives: FixedLengthArray of float
        The values of the objectives.  These values will only be assigned after
        the solution is evaluated.
    constraints: FixedLengthArray of float
        The values of the constraints.  These values will only be assigned
        after the solution is evaluated.
    constraint_violation: float
        The magnitude of the constraint violation.
    feasible: bool
        True if the solution does not violate any constraints; False otherwise.
    evaluated: bool
        True if the solution is evaluated; False otherwise.
    """

    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.variables = FixedLengthArray(problem.nvars)
        self.objectives = FixedLengthArray(problem.nobjs)
        self.constraints = FixedLengthArray(problem.nconstrs)
        self.constraint_violation = 0.0
        self.evaluated = False

    def evaluate(self):
        """Evaluates this solution."""
        self.problem(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Solution[" + ",".join(list(map(str, self.variables))) + "|" + ",".join(list(map(str, self.objectives))) + "|" + str(self.constraint_violation) + "]"

    def __deepcopy__(self, memo):
        """Overridden to avoid cloning the problem definition."""
        result = Solution(self.problem)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k != "problem":
                setattr(result, k, copy.deepcopy(v, memo))

        return result

class EvaluateSolution(Job):

    def __init__(self, solution):
        super().__init__()
        self.solution = solution

    def run(self):
        self.solution.evaluate()

class Algorithm(metaclass=ABCMeta):
    """Base class for all optimization algorithms.

    For most use cases, use the :meth:`run` method to execute an algorithm
    until the termination conditions are satisfied.  Internally, this invokes
    the :meth:`step` method to perform each iteration of the algorithm.  An
    termination conditions and callbacks are evaluated after each step.

    Parameters
    ----------
    problem : Problem
        The problem being optimized.
    evaluator : Evaluator
        The evalutor used to evaluate solutions.  If `None`, the default
        evaluator defined in :attr:`PlatypusConfig` is selected.
    log_frequency : int
        The frequency to log evaluation progress.  If `None`, the default
        log frequency defined in :attr:`PlatypusConfig` is selected.

    Attributes
    ----------
    nfe : int
        The current number of function evaluations (NFE)
    result: list or Archive
        The current result, which is updated after each iteration.
    """

    def __init__(self,
                 problem,
                 evaluator=None,
                 log_frequency=None,
                 **kwargs):
        super().__init__()
        self.problem = problem
        self.evaluator = evaluator
        self.nfe = 0
        self._extensions = []

        if self.evaluator is None:
            self.evaluator = PlatypusConfig.default_evaluator

        self.add_extension(PlatypusConfig.get_logging_extension(log_frequency))

    def add_extension(self, extension):
        """Adds an extension.

        Extensions add functionality to an algorithm at specific points during
        a run.  If multiple extensions are added, they are run in reverse
        order.  That is, the last extension added is the first to run.

        Parameters
        ----------
        extension : Extension
            The extension to add.
        """
        if inspect.isclass(extension):
            extension = extension()
        self._extensions.insert(0, extension)

    def remove_extension(self, extension):
        """Removes an extension.

        Parameters
        ----------
        extension : Extension or Type
            The extension or type of extension to remove.
        """
        if inspect.isclass(extension):
            self._extensions = [x for x in self._extensions if not isinstance(x, extension)]
        else:
            self._extensions = [x for x in self._extensions if x != extension]

    @abstractmethod
    def step(self):
        """Performs one logical step of the algorithm."""
        pass

    def evaluate_all(self, solutions):
        """Evaluates all of the given solutions.

        Subclasses should prefer using this method to evaluate solutions,
        ideally providing an entire population to leverage parallelization,
        as it tracks NFE.

        Parameters
        ----------
        solutions : list of Solution
            The solutions to evaluate.
        """
        unevaluated = [s for s in solutions if not s.evaluated]

        jobs = [EvaluateSolution(s) for s in unevaluated]
        results = self.evaluator.evaluate_all(jobs)

        # if needed, update the original solution with the results
        for i, result in enumerate(results):
            if unevaluated[i] != result.solution:
                unevaluated[i].variables[:] = result.solution.variables[:]
                unevaluated[i].objectives[:] = result.solution.objectives[:]
                unevaluated[i].constraints[:] = result.solution.constraints[:]
                unevaluated[i].constraint_violation = result.solution.constraint_violation
                unevaluated[i].feasible = result.solution.feasible
                unevaluated[i].evaluated = result.solution.evaluated

        self.nfe += len(solutions)

    def run(self, condition, callback=None):
        """Runs this algorithm until the termination condition is reached.

        Parameters
        ----------
        condition : int or TerminationCondition
            The termination condition.  Providing an integer value is converted
            into the :class:`MaxEvaluations` condition.
        callback : Callable, optional
            Callback function that is invoked after every iteration.  The
            callback is passed this algorithm instance.
        """
        if isinstance(condition, int):
            condition = MaxEvaluations(condition)

        if isinstance(condition, TerminationCondition):
            condition.initialize(self)

        for extension in self._extensions:
            extension.start_run(self)

        while not condition(self):
            for extension in self._extensions:
                extension.pre_step(self)

            self.step()

            for extension in self._extensions:
                extension.post_step(self)

            if callback is not None:
                callback(self)

        for extension in self._extensions:
            extension.end_run(self)

class Dominance(metaclass=ABCMeta):
    """Compares two solutions for dominance."""

    def __init__(self):
        super().__init__()

    def __call__(self, solution1, solution2):
        return self.compare(solution1, solution2)

    @abstractmethod
    def compare(self, solution1, solution2):
        """Compare two solutions for dominance.

        Parameters
        ----------
        solution1 : Solution
            The first solution.
        solution2 : Solution
            The second solution.

        Returns
        -------
        `-1` if the first solution dominates the second, `1` if the second
        solution dominates the first, or `0` if the two solutions are mutually
        non-dominated.
        """
        raise NotImplementedError()

class ParetoDominance(Dominance):
    """Pareto dominance with constraints.

    If either solution violates constraints, then the solution with a smaller
    constraint violation is preferred.  If both solutions are feasible, then
    Pareto dominance is used to select the preferred solution.
    """

    def __init__(self):
        super().__init__()

    def compare(self, solution1, solution2):
        problem = solution1.problem

        # first check constraint violation
        if problem.nconstrs > 0 and solution1.constraint_violation != solution2.constraint_violation:
            if solution1.constraint_violation == 0.0:
                return -1
            elif solution2.constraint_violation == 0.0:
                return 1
            elif solution1.constraint_violation < solution2.constraint_violation:
                return -1
            elif solution2.constraint_violation < solution1.constraint_violation:
                return 1

        # then use Pareto dominance on the objectives
        dominate1 = False
        dominate2 = False

        for i in range(problem.nobjs):
            o1 = solution1.objectives[i]
            o2 = solution2.objectives[i]

            if problem.directions[i] == Direction.MAXIMIZE:
                o1 = -o1
                o2 = -o2

            if o1 < o2:
                dominate1 = True

                if dominate2:
                    return 0
            elif o1 > o2:
                dominate2 = True

                if dominate1:
                    return 0

        if dominate1 == dominate2:
            return 0
        elif dominate1:
            return -1
        else:
            return 1

class EpsilonDominance(Dominance):
    """Epsilon dominance.

    Similar to Pareto dominance except if the two solutions are contained
    within the same epsilon-box, the solution closer to the optimal corner
    of the box is preferred.

    Parameters
    ----------
    epsilons : list of float
        The epsilons that define the sizes of the epsilon-box.
    """

    def __init__(self, epsilons):
        super().__init__()

        if hasattr(epsilons, "__getitem__"):
            self.epsilons = epsilons
        else:
            self.epsilons = [epsilons]

    def same_box(self, solution1, solution2):
        """Determines if the two solutions exist in the same epsilon-box.

        Parameters
        ----------
        solution1 : Solution
            The first solution.
        solution2: Solution
            The second solution.

        Returns
        -------
        :code:`True` if the two solutions are in the same epsilon-box,
        :code:`False` otherwise.
        """
        problem = solution1.problem

        # first check constraint violation
        if problem.nconstrs > 0 and solution1.constraint_violation != solution2.constraint_violation:
            if solution1.constraint_violation == 0:
                return False
            elif solution2.constraint_violation == 0:
                return False
            elif solution1.constraint_violation < solution2.constraint_violation:
                return False
            elif solution2.constraint_violation < solution1.constraint_violation:
                return False

        # then use epsilon dominance on the objectives
        dominate1 = False
        dominate2 = False

        for i in range(problem.nobjs):
            o1 = solution1.objectives[i]
            o2 = solution2.objectives[i]

            if problem.directions[i] == Direction.MAXIMIZE:
                o1 = -o1
                o2 = -o2

            epsilon = float(self.epsilons[i if i < len(self.epsilons) else -1])
            i1 = math.floor(o1 / epsilon)
            i2 = math.floor(o2 / epsilon)

            if i1 < i2:
                dominate1 = True

                if dominate2:
                    return False
            elif i1 > i2:
                dominate2 = True

                if dominate1:
                    return False

        if not dominate1 and not dominate2:
            return True
        else:
            return False

    def compare(self, solution1, solution2):
        problem = solution1.problem

        # first check constraint violation
        if problem.nconstrs > 0 and solution1.constraint_violation != solution2.constraint_violation:
            if solution1.constraint_violation == 0:
                return -1
            elif solution2.constraint_violation == 0:
                return 1
            elif solution1.constraint_violation < solution2.constraint_violation:
                return -1
            elif solution2.constraint_violation < solution1.constraint_violation:
                return 1

        # then use epsilon dominance on the objectives
        dominate1 = False
        dominate2 = False

        for i in range(problem.nobjs):
            o1 = solution1.objectives[i]
            o2 = solution2.objectives[i]

            if problem.directions[i] == Direction.MAXIMIZE:
                o1 = -o1
                o2 = -o2

            epsilon = float(self.epsilons[i if i < len(self.epsilons) else -1])
            i1 = math.floor(o1 / epsilon)
            i2 = math.floor(o2 / epsilon)

            if i1 < i2:
                dominate1 = True

                if dominate2:
                    return 0
            elif i1 > i2:
                dominate2 = True

                if dominate1:
                    return 0

        if not dominate1 and not dominate2:
            dist1 = 0.0
            dist2 = 0.0

            for i in range(problem.nobjs):
                o1 = solution1.objectives[i]
                o2 = solution2.objectives[i]

                if problem.directions[i] == Direction.MAXIMIZE:
                    o1 = -o1
                    o2 = -o2

                epsilon = float(self.epsilons[i if i < len(self.epsilons) else -1])
                i1 = math.floor(o1 / epsilon)
                i2 = math.floor(o2 / epsilon)

                dist1 += math.pow(o1 - i1*epsilon, 2.0)
                dist2 += math.pow(o2 - i2*epsilon, 2.0)

            if dist1 < dist2:
                return -1
            else:
                return 1
        elif dominate1:
            return -1
        else:
            return 1

class AttributeDominance(Dominance):
    """Dominance based on the value of an attribute.

    The referenced attribute must be numeric, typically either an int or float.

    Parameters
    ----------
    getter : Callable
        Function that reads the value of the attribute from each solution.
    larger_preferred : bool
        Determines if larger or smaller values are preferred.
    """

    def __init__(self, getter, larger_preferred=True):
        super().__init__()
        self.larger_preferred = larger_preferred

        if hasattr(getter, "__call__"):
            self.getter = getter
        else:
            self.getter = operator.attrgetter(getter)

    def compare(self, solution1, solution2):
        a = self.getter(solution1)
        b = self.getter(solution2)

        if self.larger_preferred:
            a = -a
            b = -b

        if a < b:
            return -1
        elif a > b:
            return 1
        else:
            return 0

class Archive:
    """An archive containing only non-dominated solutions.

    Since an archive stores non-dominated solutions, its size can potentially
    grow unbounded.  Consider using one of the subclasses that provide
    truncation.

    Parameters
    ----------
    dominance : Dominance
        The dominance criteria (default is Pareto dominance).
    """

    def __init__(self, dominance=ParetoDominance()):
        super().__init__()
        self._dominance = dominance
        self._contents = []

    def add(self, solution):
        """Try adding a solution to this archive.

        Three outcomes can occur when adding a solution:
        1. The solution is non-dominated.  The new solution is added to the
           archive.
        2. The solution dominiates one or more members of the archive.  The
           dominated solutions are removed and the new solution added.
        3. The solution is dominated by one or more members of the archive.
           The new solution is rejected and the archive is unchanged.

        Parameters
        ----------
        solution : Solution
            The solution to add.

        Returns
        -------
        :code:`True` if the solution is non-dominated and added to the archive,
        :code:`False` otherwise.
        """
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]

        if any(dominates):
            return False
        else:
            self._contents = list(itertools.compress(self._contents, nondominated)) + [solution]
            return True

    def append(self, solution):
        """Append a solution to this archive.

        This is similar to :meth:`add` except no result is returned.

        Parameters
        ----------
        solution : Solution
            The solution to append.
        """
        self.add(solution)

    def extend(self, solutions):
        """Appends a list of solutions to this archive.

        Parameters
        ----------
        solutions : iterable of Solution
            The solutions to append.
        """
        for solution in solutions:
            self.append(solution)

    def remove(self, solution):
        """Try removing the solution from this archive.

        Parameters
        ----------
        solution : Solution
            The solution to remove.

        Returns
        -------
        :code:`True` if the solution was removed from this archive,
        :code:`False` otherwise.
        """
        try:
            self._contents.remove(solution)
            return True
        except ValueError:
            return False

    def __len__(self):
        return len(self._contents)

    def __getitem__(self, key):
        return self._contents[key]

    def __iadd__(self, other):
        if hasattr(other, "__iter__"):
            for o in other:
                self.add(o)
        else:
            self.add(other)

        return self

    def __iter__(self):
        return iter(self._contents)

class AdaptiveGridArchive(Archive):
    """A bounded archive using density to truncate solutions.

    The objective space is partitioned into a grid containing
    :code:`math.pow(divisions, nobjs)` cells.  Please note that this can
    quickly result in a large internal array or an integer overflow as either
    :code:`divisions` or :code:`nobjs` grows.

    The density of each cell is measured by counting the number of solutions
    within the cell.  When the archive exceeds the desired capacity, a solution
    is removed from the densest cell(s).

    Parameters
    ----------
    capacity : int
        The maximum capacity of this archive.
    nobjs : int
        The number of objectives.
    divisions : int
        The number of divisions in objective space
    dominance : Dominance
        The dominance criteria (default is Pareto dominance).
    """

    def __init__(self, capacity, nobjs, divisions, dominance=ParetoDominance()):
        super().__init__(dominance)
        self.capacity = capacity
        self.nobjs = nobjs
        self.divisions = divisions

        self.adapt_grid()

    def add(self, solution):
        # check if the candidate solution dominates or is dominated
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]

        if any(dominates):
            return False

        self._contents = list(itertools.compress(self._contents, nondominated))

        # archive is empty, add the candidate
        if len(self) == 0:
            self._contents.append(solution)
            self.adapt_grid()
            return True

        # temporarily add the candidate solution
        self._contents.append(solution)
        index = self.find_index(solution)

        if index < 0:
            self.adapt_grid()
            index = self.find_index(solution)
        else:
            self.density[index] += 1

        if len(self) <= self.capacity:
            # keep the candidate if size is less than capacity
            return True
        elif self.density[index] == self.density[self.find_densest()]:
            # reject candidate if in most dense cell
            self.remove(solution)
            return False
        else:
            # keep candidate and remove one from densest cell
            self.remove(self.pick_from_densest())
            return True

    def remove(self, solution):
        removed = super().remove(solution)

        if removed:
            index = self.find_index(solution)

            if self.density[index] > 1:
                self.density[index] -= 1
            else:
                self.adapt_grid()

        return removed

    def adapt_grid(self):
        """Adapts the grid by updating the bounds and density."""
        self.minimum = [POSITIVE_INFINITY]*self.nobjs
        self.maximum = [-POSITIVE_INFINITY]*self.nobjs
        self.density = [0.0]*(self.divisions**self.nobjs)

        for solution in self:
            for i in range(self.nobjs):
                self.minimum[i] = min(self.minimum[i], solution.objectives[i])
                self.maximum[i] = max(self.maximum[i], solution.objectives[i])

        for solution in self:
            self.density[self.find_index(solution)] += 1

    def find_index(self, solution):
        """Returns the grid cell index of the given solution."""
        index = 0

        for i in range(self.nobjs):
            value = solution.objectives[i]

            if value < self.minimum[i] or value > self.maximum[i]:
                return -1

            if self.maximum[i] > self.minimum[i]:
                value = (value - self.minimum[i]) / (self.maximum[i] - self.minimum[i])
            else:
                value = 0

            temp_index = int(self.divisions * value)

            if temp_index == self.divisions:
                temp_index -= 1

            index += temp_index * pow(self.divisions, i)

        return index

    def find_densest(self):
        """Finds the grid cell index with the highest density."""
        index = -1
        value = -1

        for i in range(len(self)):
            temp_index = self.find_index(self[i])
            temp_value = self.density[temp_index]

            if temp_value > value:
                value = temp_value
                index = temp_index

        return index

    def pick_from_densest(self):
        """Picks a solution from the densest grid cell(s)."""
        solution = None
        value = -1

        for i in range(len(self)):
            temp_value = self.density[self.find_index(self[i])]

            if temp_value > value:
                solution = self[i]
                value = temp_value

        return solution

class FitnessArchive(Archive):
    """A bounded archive that uses fitness to truncate solutions.

    Fitness is a generic term, simply referring to numeric attributes assigned
    to a solution.  For instance, below we use :meth:`crowding_distance` to
    assign the :code:`crowding_distance` attribute, and
    :meth:`crowding_distance_key` to read those values::

        FitnessArchive(crowding_distance, getter=crowding_distance_key)

    Refer to :meth:`truncate_fitness` for more details.

    Parameters
    ----------
    fitness : Callable
        Function for calculating and assigning a fitness attribute to all
        members of the archive.
    dominance : Dominance
        The dominance criteria (default is Pareto dominance).
    larger_preferred : bool
        Determines if larger or smaller fitness values are preferred during
        truncation.
    getter : Callable
        Function that reads the fitness attribute from a solution.  This should
        match the attribute assigned by the :code:`fitness` function.
    """

    def __init__(self, fitness, dominance=ParetoDominance(), larger_preferred=True, getter=fitness_key):
        super().__init__(dominance)
        self.fitness = fitness
        self.larger_preferred = larger_preferred
        self.getter = getter

    def truncate(self, size):
        """Truncates the archive to the given size.

        Parameters
        ----------
        size : int
            The desired size of this archive.
        """
        self.fitness(self._contents)
        self._contents = truncate_fitness(self._contents,
                                          size,
                                          larger_preferred=self.larger_preferred,
                                          getter=self.getter)

class EpsilonBoxArchive(Archive):
    """Archive based on epsilon-box dominance.

    Uses :class:`EpsilonDominance` to limit the size of the archive.  This
    avoids adding many non-dominated solutions that are similar to one
    another, as controlled by the provided :code:`epsilons`.

    Parameters
    ----------
    epsilons : list of float
        The epsilons that control the size of the epsilon boxes.

    Attributes
    ----------
    improvements : int
        Tracks the number of epsilon-box improvements, which by definition
        counts the number of new solutions accepted into this archive.
    """

    def __init__(self, epsilons):
        super().__init__(EpsilonDominance(epsilons))
        self.improvements = 0

    def add(self, solution):
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]
        # dominated = [x < 0 for x in flags]
        not_same_box = [not self._dominance.same_box(solution, s) for s in self._contents]

        if any(dominates):
            return False
        else:
            self._contents = list(itertools.compress(self._contents, nondominated)) + [solution]

            if all(not_same_box):
                self.improvements += 1

            return True

def nondominated(solutions):
    """Filters the solutions to only include non-dominated.

    Parameters
    ----------
    solutions : iterable of Solution
        The solutions to filter.

    Returns
    -------
    The non-dominated solutions.
    """
    archive = Archive()
    archive += solutions
    return archive._contents

def nondominated_sort_cmp(x, y):
    """Compares two solutions using nondominated sorting results.

    After processing a population with :func:`nondominated_sort`, this
    comparison function can be used to order solutions by their rank and
    crowding distance.

    Parameters
    ----------
    x : Solution
        The first solution.
    y : Solution
        The second solution.

    Returns
    -------
    :code:`-1`, :code:`0`, or :code:`1` to indicate if :code:`x` is better,
    equal, or worse than :code:`y` based on rank and crowding distance.
    """
    if x.rank == y.rank:
        if -x.crowding_distance < -y.crowding_distance:
            return -1
        elif -x.crowding_distance > -y.crowding_distance:
            return 1
        else:
            return 0
    else:
        if x.rank < y.rank:
            return -1
        elif x.rank > y.rank:
            return 1
        else:
            return 0

def nondominated_sort(solutions):
    """Fast non-dominated sorting.

    Performs fast non-dominated sorting on a collection of solutions.  The
    solutions will be assigned the following attributes:

    1. :code:`rank` - The index of the non-dominated front containing the
       solution.  Rank 0 stores all non-dominated solutions.

    2. :code:`crowding_distance` - The crowding distance of the given solution.
       Larger values indicate less crowding near the solution.

    Parameters
    ----------
    solutions : iterable of Solution
        The collection of solutions
    """
    rank = 0

    while len(solutions) > 0:
        archive = Archive()
        archive += solutions

        for solution in archive:
            solution.rank = rank

        crowding_distance(archive)

        solutions = [x for x in solutions if x not in archive]
        rank += 1

def crowding_distance(solutions):
    """Calculates crowding distance for a non-dominated front.

    Computes the crowding distance for a single non-dominated front.  It is
    assumed all solutions are non-dominated.  This method assigns the attribute
    :code:`crowding_distance` to all solutions.

    Parameters
    ----------
    solutions : iterable of Solution
        The collection of solutions
    """
    for solution in solutions:
        solution.crowding_distance = 0.0

    solutions = unique(solutions)

    if len(solutions) < 3:
        for solution in solutions:
            solution.crowding_distance = POSITIVE_INFINITY
    else:
        nobjs = solutions[0].problem.nobjs

        for i in range(nobjs):
            sorted_solutions = sorted(solutions, key=objective_value_at_index(i))
            min_value = sorted_solutions[0].objectives[i]
            max_value = sorted_solutions[-1].objectives[i]

            sorted_solutions[0].crowding_distance += POSITIVE_INFINITY
            sorted_solutions[-1].crowding_distance += POSITIVE_INFINITY

            for j in range(1, len(sorted_solutions)-1):
                if max_value - min_value < EPSILON:
                    sorted_solutions[j].crowding_distance = POSITIVE_INFINITY
                else:
                    diff = sorted_solutions[j+1].objectives[i] - sorted_solutions[j-1].objectives[i]
                    sorted_solutions[j].crowding_distance += diff / (max_value - min_value)

def nondominated_split(solutions, size):
    """Identify the front that must be truncated.

    When truncating a population to a fixed size using non-dominated sorting,
    identify the front N that can not completely fit within the truncated
    result.  Returns a tuple :code:`(first, last)`, where :code:`first`
    contains all solutions in the first N fronts (those solutions that will
    definitely remain after truncation), and the :code:`last` is the front
    that must be truncated.  :code:`last` can be empty.

    Parameters
    ----------
    solutions : iterable of Solution
        The collection of solutions that have been non-dominated sorted
    size : int
        The size of the truncated result
    """
    result = []
    rank = 0

    while len(result) < size:
        front = matches(solutions, rank, key=rank_key)

        if len(front) == 0:
            break

        if len(result)+len(front) <= size:
            result.extend(front)
        else:
            return (result, front)

        rank += 1

    return result, []

def nondominated_prune(solutions, size):
    """Prunes a population using non-dominated sorting.

    Similar to :code:`nondominated_truncate`, except the crowding distance
    is recomputed after each solution is removed.

    Parameters
    ----------
    solutions : iterable
        The collection of solutions that have been non-domination sorted
    size: int
        The size of the truncated result
    """
    result, remaining = nondominated_split(solutions, size)

    while len(result) + len(remaining) > size:
        crowding_distance(remaining)
        remaining = truncate(remaining, len(remaining)-1, key=crowding_distance_key, reverse=True)

    return result + remaining

def nondominated_truncate(solutions, size):
    """Truncates a population using non-dominated sorting.

    Truncates a population to the given size.  The resulting population is
    filled with the first N-1 fronts.  The Nth front is too large and must be
    split using crowding distance.

    Parameters
    ----------
    solutions : iterable
        The collection of solutions that have been non-domination sorted
    size: int
        The size of the truncated result
    """
    return truncate(solutions, size, key=functools.cmp_to_key(nondominated_sort_cmp))

def truncate_fitness(solutions, size, larger_preferred=True, getter=fitness_key):
    """Truncates a population based on a fitness value.

    Truncates a population to the given size based on fitness values.  By
    default, the attribute :code:`fitness` is used, but can be customized by
    specifying a custom :code:`attrgetter`.

    Parameters
    ----------
    solutions : iterable
        The collection of solutions that have already been assigned fitness
        values
    size : int
        The size of the truncated result
    larger_preferred : bool (default True)
        If larger fitness values are preferred
    getter : callable (default :code:`attrgetter("fitness")`)
        Retrieves the fitness value from a solution
    """
    return truncate(solutions, size, key=getter, reverse=larger_preferred)

def normalize(solutions, minimum=None, maximum=None):
    """Normalizes the solution objectives.

    Normalizes the objectives of each solution within the minimum and maximum
    bounds.  If the minimum and maximum bounds are not provided, then the
    bounds are computed based on the bounds of the solutions.

    Returns the minimum and maximum bounds used for normalization along with
    setting the 'normalized_objectives' attribute on each solution.

    Parameters
    ----------
    solutions : iterable
        The solutions to be normalized.
    minimum : float list
        The minimum values used to normalize the objectives.
    maximum : float list
        The maximum values used to normalize the objectives.
    """
    if len(solutions) == 0:
        return

    problem = solutions[0].problem
    feasible = [s for s in solutions if s.constraint_violation == 0.0]

    if minimum is None:
        minimum = [min([s.objectives[i] for s in feasible]) for i in range(problem.nobjs)]

    if maximum is None:
        maximum = [max([s.objectives[i] for s in feasible]) for i in range(problem.nobjs)]

    if any([abs(maximum[i]-minimum[i]) < EPSILON for i in range(problem.nobjs)]):
        raise PlatypusError("objective with empty range")

    for s in feasible:
        s.normalized_objectives = [(s.objectives[i] - minimum[i]) / (maximum[i] - minimum[i]) for i in range(problem.nobjs)]

    return minimum, maximum

class FitnessEvaluator(metaclass=ABCMeta):

    def __init__(self, kappa=0.05):
        super().__init__()
        self.kappa = kappa

    @abstractmethod
    def calculate_indicator(self, solution1, solution2):
        raise NotImplementedError()

    def evaluate(self, solutions):
        if len(solutions) == 0:
            return

        normalize(solutions)
        self.fitcomp = []
        self.max_fitness = -POSITIVE_INFINITY

        for i in range(len(solutions)):
            self.fitcomp.append([])

            for j in range(len(solutions)):
                value = self.calculate_indicator(solutions[i], solutions[j])
                self.fitcomp[i].append(value)
                self.max_fitness = max(self.max_fitness, abs(value))

        for i in range(len(solutions)):
            sum = 0.0

            for j in range(len(solutions)):
                if i != j:
                    sum += math.exp((-self.fitcomp[j][i] / self.max_fitness) / self.kappa)

            solutions[i].fitness = sum

    def remove(self, solutions, index):
        for i in range(len(solutions)):
            if i != index:
                fitness = solutions[i].fitness
                fitness -= math.exp((-self.fitcomp[index][i] / self.max_fitness) / self.kappa)
                solutions[i].fitness = fitness

        for i in range(len(solutions)):
            for j in range(index+1, len(solutions)):
                self.fitcomp[i][j-1] = self.fitcomp[i][j]

            if i > index:
                self.fitcomp[i-1] = self.fitcomp[i]

        del solutions[index]

class HypervolumeFitnessEvaluator(FitnessEvaluator):

    def __init__(self,
                 kappa=0.05,
                 rho=2.0,
                 dominance=ParetoDominance()):
        super().__init__(kappa=kappa)
        self.rho = rho
        self.dominance = dominance

    def calculate_indicator(self, solution1, solution2):
        problem = solution1.problem

        if self.dominance.compare(solution1, solution2) < 0:
            return -self.hypervolume(solution1, solution2, problem.nobjs)
        else:
            return self.hypervolume(solution2, solution1, problem.nobjs)

    def hypervolume(self, solution1, solution2, d):
        a = solution1.normalized_objectives[d-1]

        if solution2 is None:
            b = self.rho
        else:
            b = solution2.normalized_objectives[d-1]

        if solution1.problem.directions[d-1] == Direction.MAXIMIZE:
            a = 1.0 - a
            b = 1.0 - b

        if d == 1:
            if a < b:
                return (b - a) / self.rho
            else:
                return 0.0
        else:
            if a < b:
                return self.hypervolume(solution1, None, d-1)*(b-a)/self.rho + \
                       self.hypervolume(solution1, solution2, d-1)*(self.rho-b)/self.rho
            else:
                return self.hypervolume(solution1, solution2, d-1)*(self.rho-a)/self.rho

class Indicator(metaclass=ABCMeta):
    """Abstract class for performance indicators."""

    def __init__(self):
        super().__init__()

    def __call__(self, set):
        return self.calculate(set)

    @abstractmethod
    def calculate(self, set):
        """Calculates and returns the indicator value.

        Parameters
        ----------
        set : iterable of Solution
            The collection of solutions against which the indicator value is
            computed.
        """
        raise NotImplementedError()
