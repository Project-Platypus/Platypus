import copy
import random
from platypus.core import PlatypusError, Solution, ParetoDominance, Generator, Selector, Variator, Mutation, EPSILON
from platypus.types import Real

def clip(value, min_value, max_value):
    return max(min_value, min(value, max_value))

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
    
    def __init__(self, tournament_size = 2, dominance = ParetoDominance()):
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
                
        return winner
   
class PM(Mutation):
    
    def __init__(self, probability, distribution_index = 20.0):
        super(PM, self).__init__()
        self.probability = probability
        self.distribution_index = distribution_index
        
    def mutate(self, parent):
        child = copy.deepcopy(parent)
        problem = child.problem
        
        for i in range(len(child.variables)):
            if isinstance(problem.types[i], Real):
                if random.uniform(0.0, 1.0) <= self.probability:
                    child.variables[i] = self.pm_mutation(float(child.variables[i]),
                                                          problem.types[i].min_value,
                                                          problem.types[i].max_value)
                    child.evaluated = False
        
        return child
    
    def pm_mutation(self, x, lb, ub):
        u = random.uniform(0, 1)
        dx = ub - lb
        
        if u < 0.5:
            bl = (x - lb) / dx
            b = 2.0*u + (1.0 - 2.0*u)*pow(1.0 - bl, self.distribution_index + 1.0)
            delta = pow(b, 1.0 / (self.distribution_index + 1.0)) - 1.0
        else:
            bu = (ub - x) / dx
            b = 2.0*(1.0 - u) + 2.0*(u - 0.5)*pow(1.0 - bu, self.distribution_index + 1.0)
            delta = 1.0 - pow(b, 1.0 / (self.distribution_index + 1.0))
            
        x = x + delta*dx
        x = clip(x, lb, ub)
            
        return x
    
class SBX(Variator):
     
    def __init__(self, probability, distribution_index = 15.0):
        super(SBX, self).__init__(2)
        self.probability = probability
        self.distribution_index = distribution_index
         
    def evolve(self, parents):
        child1 = copy.deepcopy(parents[0])
        child2 = copy.deepcopy(parents[1])
        
        if random.uniform(0.0, 1.0) <= self.probability:
            problem = child1.problem
            nvars = problem.nvars
            
            for i in range(nvars):
                if isinstance(problem.types[i], Real):
                    if random.uniform(0.0, 1.0) <= 0.5:
                        x1 = float(child1.variables[i])
                        x2 = float(child2.variables[i])
                        lb = problem.types[i].min_value
                        ub = problem.types[i].max_value
                        
                        x1, x2 = self.sbx_crossover(x1, x2, lb, ub)
                        
                        child1.variables[i] = x1
                        child2.variables[i] = x2
                        child1.evaluated = False
                        child2.evaluated = False
                    
        return [child1, child2]
                    
    def sbx_crossover(self, x1, x2, lb, ub):
        dx = x2 - x1
        
        if dx > EPSILON:
            if x1 < x2:
                bl = 1.0 + 2.0*(x1 - lb) / dx
                bu = 1.0 + 2.0*(ub - x2) / dx
            else:
                bl = 1.0 + 2.0*(x2 - lb) / dx
                bu = 1.0 + 2.0*(ub - x1) / dx
                
            # use symmetric distributions
            if bl < bu:
                bu = bl
            else:
                bl = bu
                
            p_bl = 1.0 - 1.0 / (2.0 * pow(bl, self.distribution_index + 1.0))
            p_bu = 1.0 - 1.0 / (2.0 * pow(bu, self.distribution_index + 1.0))
            u = random.uniform(0.0, 1.0)
            
            if (u == 1.0):
                u -= EPSILON
                
            u1 = u * p_bl
            u2 = u * p_bu
            
            if u1 <= 0.5:
                b1 = pow(2.0 * u1, 1.0 / (self.distribution_index + 1.0))
            else:
                b1 = pow(0.5 / (1.0 - u1), 1.0 / (self.distribution_index + 1.0))
                
            if u2 <= 0.5:
                b2 = pow(2.0 * u2, 1.0 / (self.distribution_index + 1.0))
            else:
                b2 = pow(0.5 / (1.0 - u2), 1.0 / (self.distribution_index + 1.0))
                
            if x1 < x2:
                x1 = 0.5 * (x1 + x2 + b1*(x1 - x2))
                x2 = 0.5 * (x1 + x2 + b2*(x2 - x1))
            else:
                x1 = 0.5 * (x1 + x2 + b2*(x1 - x2))
                x2 = 0.5 * (x1 + x2 + b1*(x2 - x1))
                
            # randomly swap the values
            if bool(random.getrandbits(1)):
                x1, x2 = x2, x1
        
            x1 = clip(x1, lb, ub)
            x2 = clip(x2, lb, ub)
            
        return x1, x2

class GAOperator(Variator):
    
    def __init__(self, variation, mutation):
        super(GAOperator, self).__init__(variation.arity)
        self.variation = variation
        self.mutation = mutation
        
    def evolve(self, parents):
        return map(self.mutation.evolve, self.variation.evolve(parents))
    
class DifferentialEvolution(Variator):
    
    def __init__(self, crossover_rate=0.1, step_size=0.5):
        super(DifferentialEvolution, self).__init__(4)
        self.crossover_rate = crossover_rate
        self.step_size = step_size
        
    def evolve(self, parents):
        result = copy.deepcopy(parents[0])
        problem = result.problem
        jrand = random.randrange(problem.nvars)
        
        for j in range(problem.nvars):
            if random.uniform(0.0, 1.0) <= self.crossover_rate or j == jrand:
                v1 = float(parents[1].variables[j])
                v2 = float(parents[2].variables[j])
                v3 = float(parents[3].variables[j])
                
                y = v3 + self.step_size*(v1 - v2)
                y = clip(y, problem.types[j].min_value, problem.types[j].max_value)
                
                result.variables[j] = y
                result.evaluated = False
                
        return [result]