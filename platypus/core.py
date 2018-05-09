# Copyright 2015-2018 David Hadka
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
from __future__ import absolute_import, division, print_function

import sys
import copy
import math
import time
import logging
import datetime
import operator
import functools
import itertools
from abc import ABCMeta, abstractmethod
from .evaluator import Job

LOGGER = logging.getLogger("Platypus")
EPSILON = sys.float_info.epsilon
POSITIVE_INFINITY = float("inf")

class PlatypusError(Exception):
    pass

def fitness_key(x):
    return x.fitness

def crowding_distance_key(x):
    return x.crowding_distance

def objective_key(x, index=0):
    return x.objectives[index]

class FixedLengthArray(object):

    def __init__(self, size, default_value = None, convert = None):
        super(FixedLengthArray, self).__init__()
        self._size = size
        self._data = [default_value]*size
        self.convert = convert
        
    def __len__(self):
        return self._size
        
    def __setitem__(self, index, value):
        if self.convert is not None:
            value = self.convert(value)
        
        if type(index) == slice:
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
    
def _convert_constraint(x):
    if isinstance(x, Constraint):
        return x
    elif isinstance(x, (list, tuple)):
        return [_convert_constraint(y) for y in x]
    else:
        return Constraint(x)

class Problem(object):
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
    directions: FixedLengthArray of int
        The optimization direction of each objective, either MINIMIZE (-1) or
        MAXIMIZE (1)
    constraints: FixedLengthArray of Constraint
        Describes the types of constraints as an equality or inequality.  The
        default requires each constraint value to be 0.
    """
    
    MINIMIZE = -1
    MAXIMIZE = 1
    
    def __init__(self, nvars, nobjs, nconstrs = 0, function=None):
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
        super(Problem, self).__init__()
        self.nvars = nvars
        self.nobjs = nobjs
        self.nconstrs = nconstrs
        self.function = function
        self.types = FixedLengthArray(nvars)
        self.directions = FixedLengthArray(nobjs, self.MINIMIZE)
        self.constraints = FixedLengthArray(nconstrs, "==0", _convert_constraint)
        
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

class Generator(object):
    """Abstract class for generating initial populations."""
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Generator, self).__init__()
        
    @abstractmethod
    def generate(self, problem):
        raise NotImplementedError("method not implemented")
    
class Variator(object):
    """Abstract class for variation operators (crossover and mutation)."""
    
    __metaclass__ = ABCMeta
    
    def __init__(self, arity):
        super(Variator, self).__init__()
        self.arity = arity
        
    @abstractmethod
    def evolve(self, parents):
        raise NotImplementedError("method not implemented")
    
class Mutation(Variator):
    """Variator for mutation, which requires only one parent."""
    
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Mutation, self).__init__(1)
        
    def evolve(self, parents):
        if hasattr(parents, "__iter__"):
            return list(map(self.mutate, parents))
        else:
            return self.mutate(parents)
        
    @abstractmethod
    def mutate(self, parent):
        raise NotImplementedError("method not implemented")
    
class Selector(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Selector, self).__init__()
        
    def select(self, n, population):
        return list(map(self.select_one, itertools.repeat(population, n)))
        
    @abstractmethod
    def select_one(self, population):
        raise NotImplementedError("method not implemented")
    
class TerminationCondition(object):
    """Abstract class for defining termination conditions."""
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(TerminationCondition, self).__init__()
        
    def __call__(self, algorithm):
        return self.shouldTerminate(algorithm)
    
    def initialize(self, algorithm):
        """Initializes this termination condition.
        
        This method is used to collect any initial state, such as the current
        NFE or current time, needed for calculating the termination criteria.
        
        Parameters
        ----------
        algorithm : Algorithm
            The algorithm being run.
        """
        pass
        
    @abstractmethod
    def shouldTerminate(self, algorithm):
        """Checks if the algorithm should terminate.
        
        Check the termination condition, returning True if the termination
        condition is satisfied; False otherwise.  This method is called after
        each iteration of the algorithm.
        
        Parameters
        ----------
        algorithm : Algorithm
            The algorithm being run.
        """
        raise NotImplementedError("method not implemented")
    
class MaxEvaluations(TerminationCondition):
    """Termination condition based on the maximum number of function evaluations.
    
    Note that since we check the termination condition after each iteration, it
    is possible for the algorithm to exceed the max NFE.
    
    Parameters
    ----------
    nfe : int
        The maximum number of function evaluations to execute.
    """
    def __init__(self, nfe):
        super(MaxEvaluations, self).__init__()
        self.nfe = nfe
        self.starting_nfe = 0
        
    def initialize(self, algorithm):
        self.starting_nfe = algorithm.nfe
        
    def shouldTerminate(self, algorithm):
        return algorithm.nfe - self.starting_nfe >= self.nfe
    
class MaxTime(TerminationCondition):
    """Termination condition based on the maximum elapsed time."""
    
    def __init__(self, max_time):
        super(MaxTime, self).__init__()
        self.max_time = max_time
        self.start_time = time.time()
        
    def initialize(self, algorithm):
        self.start_time = time.time()
        
    def shouldTerminate(self, algorithm):
        return time.time() - self.start_time >= self.max_time
    
class _EvaluateJob(Job):

    def __init__(self, solution):
        super(_EvaluateJob, self).__init__()
        self.solution = solution
        
    def run(self):
        self.solution.evaluate()
    
class Algorithm(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self,
                 problem,
                 evaluator=None,
                 log_frequency=None,
                 **kwargs):
        super(Algorithm, self).__init__()
        self.problem = problem
        self.evaluator = evaluator
        self.log_frequency = log_frequency
        self.nfe = 0
        
        if self.evaluator is None:
            from .config import PlatypusConfig
            self.evaluator = PlatypusConfig.default_evaluator
            
        if self.log_frequency is None:
            from .config import PlatypusConfig
            self.log_frequency = PlatypusConfig.default_log_frequency
    
    @abstractmethod
    def step(self):
        raise NotImplementedError("method not implemented")
    
    def evaluate_all(self, solutions):
        unevaluated = [s for s in solutions if not s.evaluated]
        
        jobs = [_EvaluateJob(s) for s in unevaluated]
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
        
        self.nfe += len(unevaluated)
    
    def run(self, condition, callback=None):
        if isinstance(condition, int):
            condition = MaxEvaluations(condition)
            
        if isinstance(condition, TerminationCondition):
            condition.initialize(self)
            
        last_log = self.nfe
        start_time = time.time()
        
        LOGGER.log(logging.INFO, "%s starting", type(self).__name__)

        while not condition(self):
            self.step()
            
            if self.log_frequency is not None and self.nfe >= last_log + self.log_frequency:
                LOGGER.log(logging.INFO,
                           "%s running; NFE Complete: %d, Elapsed Time: %s",
                           type(self).__name__,
                           self.nfe,
                           datetime.timedelta(seconds=time.time()-start_time))

            if callback is not None:
                callback(self)
                
        LOGGER.log(logging.INFO,
                   "%s finished; Total NFE: %d, Elapsed Time: %s",
                   type(self).__name__,
                   self.nfe,
                   datetime.timedelta(seconds=time.time()-start_time))
            
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
    
class Constraint(object):
    
    OPERATORS = {
             "==" : _constraint_eq,
             "<=" : _constraint_leq,
             ">=" : _constraint_geq,
             "!=" : _constraint_neq,
             "<"  : _constraint_lt,
             ">"  : _constraint_gt,
             }
    
    EQUALS_ZERO = "==0"
    LEQ_ZERO = "<=0"
    GEQ_ZERO = ">=0"
    LESS_THAN_ZERO = "<0"
    GREATER_THAN_ZERO = ">0"
    
    def __init__(self, op, value=None):
        super(Constraint, self).__init__()
        
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
            if op[1] == '=':
                self.function = functools.partial(Constraint.OPERATORS[op[0:2]], y=float(op[2:]))
            else:
                self.function = functools.partial(Constraint.OPERATORS[op[0:1]], y=float(op[1:]))
        
    def __call__(self, value):
        return self.function(value)
        
class Solution(object):
    """Class representing a solution to a problem.
    
    Attributes
    ----------
    problem: Problem
        The problem.
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
        """Creates a new solution for the given problem."""
        super(Solution, self).__init__()
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
        
class Dominance(object):
    """Compares two solutions for dominance."""
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Dominance, self).__init__()
        
    def __call__(self, solution1, solution2):
        return self.compare(solution1, solution2)
    
    def compare(self, solution1, solution2):
        """Compare two solutions.
        
        Returns -1 if the first solution dominates the second, 1 if the
        second solution dominates the first, or 0 if the two solutions are
        mutually non-dominated.
        
        Parameters
        ----------
        solution1 : Solution
            The first solution.
        solution2 : Solution
            The second solution.
        """
        raise NotImplementedError("method not implemented")
    
class ParetoDominance(Dominance):
    """Pareto dominance with constraints.
    
    If either solution violates constraints, then the solution with a smaller
    constraint violation is preferred.  If both solutions are feasible, then
    Pareto dominance is used to select the preferred solution.
    """
    
    def __init__(self):
        super(ParetoDominance, self).__init__()
    
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
            
            if problem.directions[i] == Problem.MAXIMIZE:
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
    or the box is preferred.
    """
    
    def __init__(self, epsilons):
        super(EpsilonDominance, self).__init__()
        
        if hasattr(epsilons, "__getitem__"):
            self.epsilons = epsilons
        else:
            self.epsilons = [epsilons]
        
    def same_box(self, solution1, solution2):
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
            
            if problem.directions[i] == Problem.MAXIMIZE:
                o1 = -o1
                o2 = -o2
                
            epsilon = float(self.epsilons[i % len(self.epsilons)])
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
            
            if problem.directions[i] == Problem.MAXIMIZE:
                o1 = -o1
                o2 = -o2
                
            epsilon = float(self.epsilons[i % len(self.epsilons)])
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
            
                if problem.directions[i] == Problem.MAXIMIZE:
                    o1 = -o1
                    o2 = -o2
                
                epsilon = float(self.epsilons[i % len(self.epsilons)])
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
    
    def __init__(self, getter, larger_preferred=True):
        super(AttributeDominance, self).__init__()
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

class Archive(object):
    """An archive only containing non-dominated solutions."""
    
    def __init__(self, dominance = ParetoDominance()):
        super(Archive, self).__init__()
        self._dominance = dominance
        self._contents = []
        
    def add(self, solution):
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]
        
        if any(dominates):
            return False
        else:
            self._contents = list(itertools.compress(self._contents, nondominated)) + [solution]
            return True
        
    def append(self, solution):
        self.add(solution)
        
    def extend(self, solutions):
        for solution in solutions:
            self.append(solution)
            
    def remove(self, solution):
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
    
    def __init__(self, capacity, nobjs, divisions, dominance = ParetoDominance()):
        super(AdaptiveGridArchive, self).__init__(dominance)
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
        removed = super(AdaptiveGridArchive, self).remove(solution)
        
        if removed:
            index = self.find_index(solution)
            
            if self.density[index] > 1:
                self.density[index] -= 1
            else:
                self.adapt_grid()
                
        return removed
        
    def adapt_grid(self):
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
        solution = None
        value = -1
        
        for i in range(len(self)):
            temp_value = self.density[self.find_index(self[i])]
            
            if temp_value > value:
                solution = self[i]
                value = temp_value
                
        return solution
    
class FitnessArchive(Archive):
    
    def __init__(self, fitness, dominance = ParetoDominance(), larger_preferred=True, getter=fitness_key):
        super(FitnessArchive, self).__init__(dominance)
        self.fitness = fitness
        self.larger_preferred = larger_preferred
        self.getter = getter
        
    def truncate(self, size):
        self.fitness(self._contents)
        self._contents = truncate_fitness(self._contents,
                                          size,
                                          larger_preferred=self.larger_preferred,
                                          getter=self.getter)
        
class EpsilonBoxArchive(Archive):
    
    def __init__(self, epsilons):
        super(EpsilonBoxArchive, self).__init__(EpsilonDominance(epsilons))
        self.improvements = 0

    def add(self, solution):
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]
        dominated = [x < 0 for x in flags]
        not_same_box = [not self._dominance.same_box(solution, s) for s in self._contents]
        
        if any(dominates):
            return False
        else:
            self._contents = list(itertools.compress(self._contents, nondominated)) + [solution]
            
            if dominated and not_same_box:
                self.improvements += 1

def unique(solutions, objectives=True):
    """Returns the unique solutions.
    
    Parameters
    ----------
    solutions : list of Solution
        The list of solutions.
    objectives: bool
        If True, then only compare solutions using their objectives.  If False,
        the compare solutions using their decision variables.
    """
    unique_ids = set()
    result = []
    
    for solution in solutions:
        problem = solution.problem
        
        if objectives:
            id = tuple(solution.objectives[:])
        else:
            id = tuple([problem.types[i].decode(solution.variables[i]) for i in range(problem.nvars)])
        
        if not id in unique_ids:
            unique_ids.add(id)
            result.append(solution)
            
    return result

def nondominated(solutions):
    """Returns the non-dominated solutions."""
    archive = Archive()
    archive += solutions
    return archive._contents

def nondominated_cmp(x, y):
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
    solutions : iterable
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
    solutions : iterable
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
            sorted_solutions = sorted(solutions, key=functools.partial(objective_key, index=i))
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
    solutions : iterable
        The collection of solutions that have been non-dominated sorted
    size : int
        The size of the truncated result
    """
    result = []
    rank = 0
    
    while len(result) < size:
        front = [x for x in solutions if x.rank==rank]
        
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
        remaining = sorted(remaining, key=crowding_distance_key)
        del remaining[0]
        
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
    result = sorted(solutions, key=functools.cmp_to_key(nondominated_cmp)) 
    return result[:size]
        
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
    result = sorted(solutions, key=getter)
    
    if larger_preferred:
        result.reverse()
    
    return result[:size]

def normalize(solutions, minimum=None, maximum=None):
    """Normalizes the solution objectives.
    
    Normalizes the objectives of each solution within the minimum and maximum
    bounds.  If the minimum and maximum bounds are not provided, then the
    bounds are computed based on the bounds of the solutions.
    
    Parameters
    ----------
    solutions : iterable
        The solutions to be normalized.
    minimum : int list
        The minimum values used to normalize the objectives.
    maximum : int list
        The maximum values used to normalize the objectives.
    """
    if len(solutions) == 0:
        return
    
    problem = solutions[0].problem
    feasible = [s for s in solutions if s.constraint_violation == 0.0]
    
    if minimum is None or maximum is None:
        if minimum is None:
            minimum = [min([s.objectives[i] for s in feasible]) for i in range(problem.nobjs)]
        
        if maximum is None:
            maximum = [max([s.objectives[i] for s in feasible]) for i in range(problem.nobjs)]
    
    if any([maximum[i]-minimum[i] < EPSILON for i in range(problem.nobjs)]):
        raise PlatypusError("objective with empty range")

    for s in feasible:
        s.normalized_objectives = [(s.objectives[i] - minimum[i]) / (maximum[i] - minimum[i]) for i in range(problem.nobjs)]
        
    return minimum, maximum

class FitnessEvaluator(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, kappa = 0.05):
        super(FitnessEvaluator, self).__init__()
        self.kappa = kappa
    
    @abstractmethod
    def calculate_indicator(self, solution1, solution2):
        raise NotImplementedError("method not implemented")
        
    def evaluate(self, solutions):
        if len(solutions) == 0:
            return
        
        normalize(solutions)
        problem = solutions[0].problem      
        self.fitcomp = []
        self.max_fitness = -POSITIVE_INFINITY
        
        for i in range(len(solutions)):
            self.fitcomp.append([])
            
            for j in range(len(solutions)):
                value  = self.calculate_indicator(solutions[i], solutions[j])
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
                 kappa = 0.05,
                 rho = 2.0,
                 dominance = ParetoDominance()):
        super(HypervolumeFitnessEvaluator, self).__init__(kappa = kappa)
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
            
        if solution1.problem.directions[d-1] == Problem.MAXIMIZE:
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

class Indicator(object):
    
    __metaclass = ABCMeta
    
    def __init__(self):
        super(Indicator, self).__init__()
        
    def __call__(self, set):
        return self.calculate(set)
        
    def calculate(self, set):
        raise NotImplementedError("method not implemented")
