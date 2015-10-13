class PlatypusError(Exception):
    pass

class FixedLengthArray(object):

    def __init__(self, size, default_value = None):
        super(FixedLengthArray, self).__init__()
        self._size = size
        self._data = [default_value]*size
        
    def __setitem__(self, index, value):
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
        self.nobjs = nobjs
        self.nconstrs = nconstrs
        self.function = function
        self.types = FixedLengthArray(nvars)
        self.directions = FixedLengthArray(nobjs, self.MINIMIZE)
        self.constraints = FixedLengthArray(nconstrs, "==0")
        
    def evaluate(self, solution):
        if self.nconstrs > 0:
            (objs, constrs) = self.function(solution)
        else:
            objs = self.function(solution)
            constrs = None
            
        if len(objs) != self.nobjs:
            raise PlatypusError("incorrect number of objectives: expected %d, received %d" % (self.nobjs, len(objs)))
        
        solution.objectives = objs
        solution.constraints = constrs
        
class Solution(object):
    
    def __init__(self, problem):
        super(Solution, self).__init__()
        self.problem = problem
        self.variables = FixedLengthArray(problem.nvars)
        self.objectives = FixedLengthArray(problem.nobjs)
        self.constraints = FixedLengthArray(problem.nconstrs)

class Archive():