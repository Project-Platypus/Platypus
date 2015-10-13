from platypus.core import Algorithm, Operator, nondominated_sort, truncate
from platypus.operators import TournamentSelector, RandomGenerator
            
class GeneticAlgorithm(Algorithm):
    
    def __init__(self, problem,
                 population_size = 100,
                 generator = RandomGenerator()):
        super(GeneticAlgorithm, self).__init__(problem)
        self.population_size = population_size
        self.generator = generator
        
    def step(self):
        if self.nfe == 0:
            self.initialize()
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
                 variator = Operator(2)):
        super(NSGAII, self).__init__(problem)
        self.population_size = population_size
        self.generator = generator
        self.selector = selector
        self.variator = variator
        
    def iterate(self):
        offspring = []
        
        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.append(self.variator.evolve(parents))
            
        self.evaluateAll(offspring)
        
        offspring.append(self.population)
        nondominated_sort(offspring)
        self.population = truncate(offspring)
            