class PlatypusError(Exception):
   pass

class Problem(object):
    
    def __init__(self, nobjs, nconstrs = 0):
        super(Problem, self).__init__()
        self.nobjs = nobjs
        self.nconstrs = nconstrs
        self.variables = None
        self.function = None
        
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
    
    def __init__(self):
        super(Solution, self).__init__()
        self.variables = None
        self.objectives = None
        self.constraints = None
        
    