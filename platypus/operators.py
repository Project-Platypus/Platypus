import random
from platypus.core import PlatypusError, Solution, ParetoDominance, Generator, Selector, Operator, Mutation
from platypus.types import Real

class RandomGenerator(Generator):
    
    def __init__(self):
        super(RandomGenerator, self).__init__()
    
    def generate(self, problem):
        solution = Solution(problem)
        solution.variables = [self.create_type(x) for x in problem.types]
        return solution
        
    def create_type(self, variable_type):
        if isinstance(variable_type, Real):
            return random.uniform(variable_type.min_value, variable_type.max_value)
        else:
            raise PlatypusError("Type %s not supported" % type(variable_type))

class TournamentSelector(Selector):
    
    def __init__(self, tournament_size = 2, dominance = ParetoDominance):
        super(TournamentSelector, self).__init__()
        self.tournament_size = tournament_size
        self.dominance = dominance
    
    def select_one(self, population):
        winner = random.choice(population)
        
        for _ in range(self.tournament_size-1):
            candidate = random.choice(population)
            flag = self.dominance.compare(winner, candidate)
            
            if flag > 0:
                winner = candidate
   
class PM(Mutation):
    
    def __init__(self, probability, distributionIndex = 15.0):
        super(PM, self).__init__()
        self.probability = probability
        self.distributionIndex = distributionIndex
        
    def mutate(self, parent):
        for i in range(len(parent.variables)):
            if isinstance(parent.problem.types[i], Real):
                parent.variables[i] = self.pm_mutation(float(parent.variables[i]),
                                                       parent.problem.types[i].min_value,
                                                       parent.problem.types[i].max_value)
    
    def pm_mutation(self, x, lb, ub):
        if random.uniform(0, 1) < self.probability:
            u = random.uniform(0, 1)
            dx = type.max - type.min
        
            if u < 0.5:
                bl = (x - type.min) / dx
                b = 2.0*u + (1.0 - 2.0*u)*pow(1.0 - bl, self.distributionIndex + 1.0)
                delta = pow(b, 1.0 / (self.distributionIndex + 1.0)) - 1.0
            else:
                bu = (type.max - x) / dx
                b = 2.0*(1.0 - u) + 2.0*(u - 0.5)*pow(1.0 - bu, self.distributionIndex + 1.0)
                delta = 1.0 - pow(b, 1.0 / (self.distributionIndex + 1.0))
            
            x = x + delta*dx
            
            if x < type.min:
                x = type.min
            if x > type.max:
                x = type.max
            
        return x
    
# class SBX(Operator):
#     
#     def __init__(self, probability, distributionIndex = 15.0):
#         super(PM, self).__init__("Simulated Binary Crossover")
#         self.probability = probability
#         self.distributionIndex = distributionIndex
#         
#     def evolve(self, parents):
#         nvars = parents[0].problem.nvars
#         
#         for i in range(len(nvars)):
#             if isinstance(parents[0].problem.types[i], Real):
#                 parent.variables[i] = self.pm_mutation(float(parent.variables[i]),
#                                                        parent.problem.types[i].min_value,
#                                                        parent.problem.types[i].max_value)
# 
#         
#         if not isinstance(type, Real):
#             raise PlatypusError("%s requires Real types" % self.name)
#         
#         x = float(value)
#         
#         if random.uniform(0, 1) < self.probability:
#             u = random.uniform(0, 1)
#             dx = type.max - type.min
#         
#             if u < 0.5:
#                 bl = (x - type.min) / dx
#                 b = 2.0*u + (1.0 - 2.0*u)*pow(1.0 - bl, self.distributionIndex + 1.0)
#                 delta = pow(b, 1.0 / (self.distributionIndex + 1.0)) - 1.0
#             else:
#                 bu = (type.max - x) / dx
#                 b = 2.0*(1.0 - u) + 2.0*(u - 0.5)*pow(1.0 - bu, self.distributionIndex + 1.0)
#                 delta = 1.0 - pow(b, 1.0 / (self.distributionIndex + 1.0))
#             
#             x = x + delta*dx
#             
#             if x < type.min:
#                 x = type.min
#             if x > type.max:
#                 x = type.max
#             
#         return x