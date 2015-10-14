import sys
import operator
import functools
import itertools
from abc import ABCMeta, abstractmethod

EPSILON = sys.float_info.epsilon
POSITIVE_INFINITY = float("inf")

class PlatypusError(Exception):
    pass

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
            if index < 0 or index >= self._size:
                raise ValueError("index is out of bounds")
            
            self._data[index] = value
            
    def __getitem__(self, index):
        return self._data[index]
    
    def __str__(self):
        return "[" + ", ".join(map(str, self._data)) + "]"

class Problem(object):
    
    MINIMIZE = -1
    MAXIMIZE = 1
    
    def __init__(self, nvars, nobjs, nconstrs = 0, function=None):
        super(Problem, self).__init__()
        self.nvars = nvars
        self.nobjs = nobjs
        self.nconstrs = nconstrs
        self.function = function
        self.types = FixedLengthArray(nvars)
        self.directions = FixedLengthArray(nobjs, self.MINIMIZE)
        self.constraints = FixedLengthArray(nconstrs, "==0", lambda x : Constraint(x))
        
    def evaluate(self, solution):
        if self.nconstrs > 0:
            (objs, constrs) = self.function(solution.variables)
        else:
            objs = self.function(solution.variables)
            constrs = []
            
        if len(objs) != self.nobjs:
            raise PlatypusError("incorrect number of objectives: expected %d, received %d" % (self.nobjs, len(objs)))
        
        if len(constrs) != self.nconstrs:
            raise PlatypusError("incorrect number of constraints: expected %d, received %d" % (self.nconstrs, len(constrs)))
        
        solution.objectives = objs
        solution.constraints = constrs
        solution.constraint_violation = sum([abs(f(x)) for (f, x) in zip(self.constraints, constrs)])
        solution.evaluated = True

class Generator(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Generator, self).__init__()
        
    @abstractmethod
    def generate(self, problem):
        raise NotImplementedError("method not implemented")
    
class Variator(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, arity):
        super(Variator, self).__init__()
        self.arity = arity
        
    @abstractmethod
    def evolve(self, parents):
        raise NotImplementedError("method not implemented")
    
class Mutation(Variator):
    
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Mutation, self).__init__(1)
        
    def evolve(self, parents):
        if hasattr(parents, "__iter__"):
            return map(self.mutate, parents)
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
        return map(self.select_one, itertools.repeat(population, n))
        
    @abstractmethod
    def select_one(self, population):
        raise NotImplementedError("method not implemented")
    
class Algorithm(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, problem):
        super(Algorithm, self).__init__()
        self.problem = problem
        self.nfe = 0
    
    @abstractmethod
    def step(self):
        raise NotImplementedError("method not implemented")
    
    def evaluateAll(self, solutions):
        unevaluated = [s for s in solutions if not s.evaluated]
        map(self.problem.evaluate, unevaluated)
        self.nfe += len(unevaluated)
    
    def run(self, NFE):
        start_nfe = self.nfe
        
        while self.nfe - start_nfe < NFE:
            self.step()
    
class Constraint(object):
    
    DELTA = 0.0001
    
    OPERATORS = {
             "==" : lambda x, y : abs(x - y),
             "<=" : lambda x, y : 0 if x <= y else abs(x - y),
             ">=" : lambda x, y : 0 if x >= y else abs(x - y),
             "!=" : lambda x, y : 0 if x != y else 1,
             "<"  : lambda x, y : 0 if x < y else abs(x - y) + Constraint.DELTA,
             ">"  : lambda x, y : 0 if x > y else abs(x - y) + Constraint.DELTA,
             }
    
    EQUALS_ZERO = "==0"
    LEQ_ZERO = "<=0"
    GEQ_ZERO = ">=0"
    LESS_THAN_ZERO = "<0"
    GREATER_THAN_ZERO = ">0"
    
    def __init__(self, op):
        super(Constraint, self).__init__()
        
        if isinstance(op, Constraint):
            self.op = op.op
            self.function = op.function
        else:
            self.op = op
            self.function = Constraint.parse(op)
        
    def __call__(self, value):
        return self.function(value)
    
    @staticmethod
    def parse(constraint):
        if hasattr(constraint, "__call__"):
            return constraint
        if constraint[1] == '=':
            return functools.partial(Constraint.OPERATORS[constraint[0:2]],
                                     y=float(constraint[2:]))
        else:
            return functools.partial(Constraint.OPERATORS[constraint[0:1]],
                                     y=float(constraint[1:]))
        
class Solution(object):
    
    def __init__(self, problem):
        super(Solution, self).__init__()
        self.problem = problem
        self.variables = FixedLengthArray(problem.nvars)
        self.objectives = FixedLengthArray(problem.nobjs)
        self.constraints = FixedLengthArray(problem.nconstrs)
        self.constraint_violation = 0.0
        self.evaluated = False
        
    def evaluate(self):
        self.problem.evaluate(self)
        
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "Solution[" + ",".join(map(str, self.objectives)) + "]"
        
class Dominance(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Dominance, self).__init__()
    
    def compare(self, solution1, solution2):
        raise NotImplementedError("method not implemented")
    
class ParetoDominance(Dominance):
    
    def __init__(self):
        super(ParetoDominance, self).__init__()
    
    def compare(self, solution1, solution2):
        if solution1.constraint_violation != solution2.constraint_violation:
            if solution1.constraint_violation == 0:
                return -1
            elif solution2.constraint_violation == 0:
                return -1
            elif solution1.constraint_violation < solution2.constraint_violation:
                return -1
            elif solution2.constraint_violation < solution1.constraint_violation:
                return 1
        
        dominates1 = any(itertools.starmap(operator.lt, itertools.izip(solution1.objectives, solution2.objectives)))
        dominates2 = any(itertools.starmap(operator.gt, itertools.izip(solution1.objectives, solution2.objectives)))
        
        if dominates1 == dominates2:
            return 0
        elif dominates1:
            return -1
        else:
            return 1
        
class AttributeDominance(Dominance):
    
    def __init__(self, getter):
        super(AttributeDominance, self).__init__()
        
        if hasattr(getter, "__call__"):
            self.getter = getter
        else:
            self.getter = operator.attrgetter(getter)
        
    def compare(self, solution1, solution2):
        return cmp(self.getter(solution1), self.getter(solution2))

class Archive(object):
    
    def __init__(self, dominance = ParetoDominance()):
        super(Archive, self).__init__()
        self._dominance = dominance
        self._contents = []
        
    def add(self, solution):
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = map(lambda x : x > 0, flags)
        nondominated = map(lambda x : x <= 0, flags)
        
        if any(dominates):
            return
        else:
            self._contents = list(itertools.compress(self._contents, nondominated)) + [solution]
    
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
    if len(solutions) < 3:
        for solution in solutions:
            solution.crowding_distance = POSITIVE_INFINITY
    else:
        nobjs = solutions[0].problem.nobjs
        
        for solution in solutions:
            solution.crowding_distance = 0.0
            
        for i in range(nobjs):
            sorted_solutions = sorted(solutions, key=lambda x : x.objectives[i])
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
        
        if len(result)+len(front) <= size:
            result.extend(front)
        elif len(front) == 0:
            return (result, [])
        else:
            return (result, front)
        
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
        remaining = sorted(remaining, key=operator.attrgetter("crowding_distance"))
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
    def comparator(x, y):
        if x.rank == y.rank:
            return cmp(-x.crowding_distance, -y.crowding_distance)
        else:
            return cmp(x.rank, y.rank)
    
    result = sorted(solutions, cmp=comparator) 
    return result[0:size]
    