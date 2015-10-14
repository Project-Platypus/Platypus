import operator
import functools
import itertools

class PlatypusError(Exception):
    pass

class FixedLengthArray(object):

    def __init__(self, size, default_value = None, convert = None):
        super(FixedLengthArray, self).__init__()
        self._size = size
        self._data = [default_value]*size
        self.convert = convert
        
    def __setitem__(self, index, value):
        if self.convert is not None:
            value = self.convert(value)
        
        if type(index) == slice:
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
    
    def __init__(self):
        super(Generator, self).__init__()
        
    def generate(self, problem):
        raise NotImplementedError("method not implemented")
    
class Operator(object):
    
    def __init__(self, arity):
        super(Operator, self).__init__()
        self.arity = arity
        
    def evolve(self, parents):
        raise NotImplementedError("method not implemented")
    
class Mutation(Operator):
    
    def __init__(self):
        super(Mutation, self).__init__(1)
        
    def evolve(self, parents):
        return map(self.mutate, parents)
        
    def mutate(self, parent):
        raise NotImplementedError("method not implemented")
    
class Selector(object):
    
    def __init__(self):
        super(Selector, self).__init__()
        
    def select(self, n, population):
        return map(self.select_one, itertools.repeat(population, n))
        
    def select_one(self, population):
        raise NotImplementedError("method not implemented")
    
class Algorithm(object):
    
    def __init__(self, problem):
        super(Algorithm, self).__init__()
        self.problem = problem
        self.nfe = 0
    
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
        self.evaluated = False
        
    def evaluate(self):
        self.problem.evaluate(self)
        
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return "Solution[" + ",".join(map(str, self.objectives)) + ";rank=" + str(self.rank) + ";crowding=" + str(self.crowding_distance) + "]"
        
class Dominance(object):
    
    def __init__(self):
        super(Dominance, self).__init__()
    
    def compare(self, solution1, solution2):
        raise NotImplementedError("method not implemented")
    
class ParetoDominance(object):
    
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

class Archive(object):
    
    def __init__(self, dominance = ParetoDominance()):
        super(Archive, self).__init__()
        self._dominance = dominance
        self._contents = []
        
    def add(self, solution):
        flags = list(itertools.starmap(self._dominance.compare, itertools.izip(itertools.cycle([solution]), self._contents)))
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
    if len(solutions) < 3:
        for solution in solutions:
            solution.crowding_distance = float("inf")
    else:
        nobjs = solutions[0].problem.nobjs
        
        for solution in solutions:
            solution.crowding_distance = 0.0
            
        for i in range(nobjs):
            sorted_solutions = sorted(solutions, key=lambda x : x.objectives[i])
            min_value = sorted_solutions[0].objectives[i]
            max_value = sorted_solutions[-1].objectives[i]
            
            sorted_solutions[0].crowding_distance += float("inf")
            sorted_solutions[-1].crowding_distance += float("inf")
            
            for j in range(1, len(sorted_solutions)-1):
                diff = sorted_solutions[j+1].objectives[i] - sorted_solutions[j-1].objectives[i]
                sorted_solutions[j].crowding_distance += diff / (max_value - min_value)

def truncate(solutions,
             size,
             primary_key=operator.attrgetter("rank"),
             secondary_key=operator.attrgetter("crowding_distance")):
    result = sorted(solutions, cmp=lambda x, y : cmp(primary_key(x), primary_key(y)) if primary_key(x)!=primary_key(y) else cmp(-secondary_key(x), -secondary_key(y)))
    return result[0:size]
    