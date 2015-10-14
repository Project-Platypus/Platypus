import math
import itertools
from platypus.core import Solution, Problem

def euclidean_dist(solution1, solution2):
    o1 = solution1.objectives
    o2 = solution2.objectives
    return math.sqrt(sum([math.pow(o1[i]-o2[i], 2.0) for i in range(len(o1))]))

class DistanceMatrix(object):
    
    def __init__(self, solutions, distance_fun=euclidean_dist):
        super(DistanceMatrix, self).__init__()
        self._solutions = solutions
        self._distance_fun = distance_fun
        
        keys = list(itertools.combinations(range(len(solutions)), 2))
        distances = map(distance_fun, [solutions[x[0]] for x in keys], [solutions[x[1]] for x in keys])
        
        self._map = {}
        for key, distance in zip(keys, distances):
            self._map[key] = distance
            
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            x = key[0]
            y = key[1]
            
            if x == y:
                return 0.0
            if y < x:
                x, y = y, x
                
            return self._map[(x, y)]
        else:
            raise ValueError("key must be a tuple")

problem = Problem(0, 2)

def createSolution(objs):
    s = Solution(problem)
    s.objectives[:] = objs
    return s
        
d = DistanceMatrix([createSolution([0, 1]), createSolution([1, 0]), createSolution([0.5, 0.5]), createSolution([0.5, 0.5])])
print d[0,1]
print d[0,2]
print d[0,3]
print d[1,2]
print d[1,3]
print d[2,3]
print d[1,0]
print d[2,0]
print d[3,0]
print d[2,1]
print d[3,1]
print d[3,2]