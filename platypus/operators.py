import random
from platypus.core import PlatypusError
from platypus.types import Real

class Operator(object):
    
    def __init__(self, name):
        super(Operator, self).__init__()
        self.name = name
        
    def evolve(self, type, value):
        raise NotImplementedError("Method not implemented")
        
class PM(Operator):
    
    def __init__(self, probability, distributionIndex = 15.0):
        super(PM, self).__init__("Polynomial Mutation")
        self.probability = probability
        self.distributionIndex = distributionIndex
        
    def evolve(self, type, value):
        if not isinstance(type, Real):
            raise PlatypusError("%s requires Real types" % self.name)
        
        x = float(value)
        
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
    
class SBX(Operator):
    
    def __init__(self, probability, distributionIndex = 15.0):
        super(PM, self).__init__("Simulated Binary Crossover")
        self.probability = probability
        self.distributionIndex = distributionIndex
        
    def evolve(self, type, value):
        if not isinstance(type, Real):
            raise PlatypusError("%s requires Real types" % self.name)
        
        x = float(value)
        
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