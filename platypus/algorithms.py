import math
import random
import operator
import itertools
from abc import ABCMeta
from platypus.core import Algorithm, Variator, Dominance, ParetoDominance, AttributeDominance, nondominated_sort, nondominated_prune, nondominated_truncate
from platypus.operators import TournamentSelector, RandomGenerator, DifferentialEvolution
            
class GeneticAlgorithm(Algorithm):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, problem,
                 population_size = 100,
                 generator = RandomGenerator()):
        super(GeneticAlgorithm, self).__init__(problem)
        self.population_size = population_size
        self.generator = generator
        self.result = []
        
    def step(self):
        if self.nfe == 0:
            self.initialize()
            self.result = self.population
        else:
            self.iterate()
            self.result = self.population
            
    def initialize(self):
        self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]
        self.evaluateAll(self.population)
            
    def iterate(self):
        raise NotImplementedError("method not implemented")
    
class NSGAII(GeneticAlgorithm):
    
    def __init__(self, problem,
                 population_size = 100,
                 generator = RandomGenerator(),
                 selector = TournamentSelector(2),
                 variator = None):
        super(NSGAII, self).__init__(problem, generator)
        self.population_size = population_size
        self.selector = selector
        self.variator = variator
        
    def iterate(self):
        offspring = []
        
        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))
            
        self.evaluateAll(offspring)
        
        offspring.extend(self.population)
        nondominated_sort(offspring)
        self.population = nondominated_truncate(offspring, self.population_size)

class GDE3(GeneticAlgorithm):
    
    def __init__(self, problem,
                 population_size = 100,
                 generator = RandomGenerator(),
                 variator = DifferentialEvolution()):
        super(GDE3, self).__init__(problem, generator)
        self.population_size = population_size
        self.variator = variator
        self.dominance = ParetoDominance()
        
    def select(self, i, arity):
        indices = []
        indices.append(i)
        indices.extend(random.sample(range(0, i) + range(i+1, len(self.population)),
                                     arity-1))
        return operator.itemgetter(*indices)(self.population)
    
    def survival(self, offspring):
        next_population = []
        
        for i in range(self.population_size):
            flag = self.dominance.compare(offspring[i], self.population[i])
            
            if flag <= 0:
                next_population.append(offspring[i])
                
            if flag >= 0:
                next_population.append(self.population[i])
                
        nondominated_sort(next_population)
        return nondominated_prune(next_population, self.population_size)    
           
    def iterate(self):
        offspring = []
        
        for i in range(self.population_size):
            parents = self.select(i, self.variator.arity)
            offspring.extend(self.variator.evolve(parents))
            
        self.evaluateAll(offspring)
        self.population = self.survival(offspring)
        
class SPEA2(GeneticAlgorithm):
    
    def __init__(self, problem,
                 population_size = 100,
                 generator = RandomGenerator(),
                 variator = None,
                 dominance = ParetoDominance(),
                 k = 1):
        super(SPEA2, self).__init__(problem, population_size, generator)
        self.variator = variator
        self.dominance = dominance
        self.k = k
        self.selection = TournamentSelector(2, dominance=AttributeDominance("fitness"))
        
    def _distance(self, solution1, solution2):
        return math.sqrt(sum([math.pow(solution2.objectives[i]-solution1.objectives[i], 2.0) for i in range(self.problem.nobjs)]))
        
    def assign_fitness(self, solutions):
        strength = [0]*len(solutions)
        fitness = [0.0]*len(solutions)
        
        # precompute dominance flags
        combinations = itertools.combinations(solutions, 2)
        flags = map(self.dominance.compare, combinations)
        distances = map(self._distance, combinations)
        
        # count the number of individuals each solution dominates
        for i in range(len(solutions)-1):
            for j in range(i+1, len(solutions)):
                flag = flags[i*len(solutions)+j]

                if flag < 0:
                    strength[i] += 1
                elif flag > 0:
                    strength[j] += 1
                    
        # the raw fitness is the sum of the dominance counts (strength) of all
        # dominated solutions
        for i in range(len(solutions)-1):
            for j in range(i+1, len(solutions)):
                flag = flags[i*len(solutions)+j]
                
                if flag < 0:
                    fitness[j] += strength[i]
                elif flag > 0:
                    fitness[i] += strength[j]
                    
        # add density to fitness
        for i in range(len(solutions)):
            distance = []
            
            for j in range(len(solutions)):
                if j < i:
                    distance.append(distances[j*len(solutions)+i])
                else:
                    distance.append(distances[i*len(solutions)+j])
                
            distance = sorted(distance)
            fitness[i] += 1.0 / (distance[self.k] + 2.0)  
            
        # assign fitness attribute
        for i in range(len(solutions)):    
            solutions[i].fitness = fitness[i]
            
    def truncate(self, solutions, size):
        survivors = [s for s in solutions if s.fitness < 1.0]
        
        if len(survivors) < size:
            solutions = sorted(solutions, key=operator.attrgetter("fitness"))
            survivors.extend(solutions[0:(size-len(survivors))])
        else:
            
            
        
    def iterate(self):
        offspring = []
        
        while len(offspring) < self.population_size:
            parents = self.selection.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))
            
        self.evaluateAll(offspring)
        
        offspring.extend(self.population)
        self.assign_fitness(offspring)
        
        self.population = truncate(offspring, self.population_size)
        