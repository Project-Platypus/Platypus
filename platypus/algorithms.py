import math
import random
import operator
import itertools
from abc import ABCMeta, abstractmethod
from platypus.core import Algorithm, Variator, Dominance, ParetoDominance, AttributeDominance, nondominated_sort, nondominated_prune, nondominated_truncate, POSITIVE_INFINITY
from platypus.operators import TournamentSelector, RandomGenerator, DifferentialEvolution
from platypus.tools import DistanceMatrix
            
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
    
    @abstractmethod
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
         
    def _assign_fitness(self, solutions):
        strength = [0]*len(solutions)
        fitness = [0.0]*len(solutions)
         
        # compute dominance flags
        keys = list(itertools.combinations(range(len(solutions)), 2))
        flags = map(self.dominance.compare, [solutions[k[0]] for k in keys], [solutions[k[1]] for k in keys])
        
        # compute the distance matrix
        distanceMatrix = DistanceMatrix(solutions)
         
        # count the number of individuals each solution dominates
        for key, flag in zip(keys, flags):
            if flag < 0:
                strength[key[0]] += 1
            elif flag > 0:
                strength[key[1]] += 1
                     
        # the raw fitness is the sum of the dominance counts (strength) of all
        # dominated solutions
        for key, flag in zip(keys, flags):
            if flag < 0:
                fitness[key[1]] += strength[key[0]]
            elif flag > 0:
                fitness[key[0]] += strength[key[1]]
                     
        # add density to fitness
        for i in range(len(solutions)):
            fitness[i] += 1.0 / (distanceMatrix.kth_distance(i, self.k) + 2.0)
             
        # assign fitness attribute
        for i in range(len(solutions)):    
            solutions[i].fitness = fitness[i]
             
    def _truncate(self, solutions, size):
        survivors = [s for s in solutions if s.fitness < 1.0]
        
        if len(survivors) < size:
            remaining = [s for s in solutions if s.fitness >= 1.0]
            remaining = sorted(remaining, key=operator.attrgetter("fitness"))
            survivors.extend(remaining[:(size-len(survivors))])
        else:
            distanceMatrix = DistanceMatrix(survivors)
            
            while len(survivors) > size:
                most_crowded = distanceMatrix.find_most_crowded()
                distanceMatrix.remove_point(most_crowded)
                del survivors[most_crowded]
                
        return survivors
    
    def initialize(self):
        super(SPEA2, self).initialize()
        self._assign_fitness(self.population)
         
    def iterate(self):
        offspring = []
         
        while len(offspring) < self.population_size:
            parents = self.selection.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))
             
        self.evaluateAll(offspring)
         
        offspring.extend(self.population)
        self._assign_fitness(offspring)
        self.population = self._truncate(offspring, self.population_size)
        
def random_weights(size, nobjs):
    weights = []
    
    if nobjs == 2:
        weights = [[1, 0], [0, 1]]
        weights.extend([(i/(size-1.0), 1.0-i/(size-1.0)) for i in range(1, size-1)])
    else:
        # generate candidate weights
        candidate_weights = []
        
        for i in range(size*50):
            random_values = [random.uniform(0.0, 1.0) for _ in range(nobjs)]
            candidate_weights.append([x/sum(random_values) for x in random_values])
        
        # add weights for the corners
        for i in range(nobjs):
            weights.append([0]*i + [1] + [0]*(nobjs-i-1))
            
        # iteratively fill in the remaining weights by finding the candidate
        # weight with the largest distance from the assigned weights
        while len(weights) < size:
            max_index = -1
            max_distance = -POSITIVE_INFINITY
            
            for i in range(len(candidate_weights)):
                distance = POSITIVE_INFINITY
                
                for j in range(len(weights)):
                    temp = math.sqrt(sum([math.pow(candidate_weights[i][k]-weights[j][k], 2.0) for k in range(nobjs)]))
                    distance = min(distance, temp)
                    
                if distance > max_distance:
                    max_index = i
                    max_distance = distance
                    
            weights.append(candidate_weights[max_index])
            del candidate_weights[max_index]
            
    return weights
    
def chebyshev(values, weights):
    return max([max(weights[i], 0.0001) * values[i] for i in range(len(values))])

class MOEAD(GeneticAlgorithm):
    
    def __init__(self, problem,
                 population_size = 100,
                 neighborhood_size = 10,
                 generator = RandomGenerator(),
                 variator = None,
                 delta = 0.8,
                 eta = 1,
                 update_utility = None,
                 weight_generator = random_weights,
                 scalarizing_function = chebyshev):
        super(MOEAD, self).__init__(problem)
        self.population_size = population_size
        self.neighborhood_size = neighborhood_size
        self.generator = generator
        self.variator = variator
        self.delta = delta
        self.eta = eta
        self.update_utility = update_utility
        self.weight_generator = weight_generator
        self.scalarizing_function = scalarizing_function
        self.generation = 0
        
    def _update_ideal(self, solution):
        for i in range(self.problem.nobjs):
            self.ideal_point[i] = min(self.ideal_point[i], solution.objectives[i])
    
    def _calculate_fitness(self, solution, weights):
        objs = solution.objectives
        normalized_objs = [objs[i]-self.ideal_point[i] for i in range(self.problem.nobjs)]
        return self.scalarizing_function(normalized_objs, weights)
    
    def _update_solution(self, solution, mating_indices):
        c = 0
        random.shuffle(mating_indices)
        
        for i in mating_indices:
            candidate = self.population[i]
            weights = self.weights[i]
            replace = False
            
            if solution.constraint_violation > 0.0 and candidate.constraint_violation > 0.0:
                if solution.constraint_violation < candidate.constraint_violation:
                    replace = True
            elif candidate.constraint_violation > 0.0:
                replace = True
            elif solution.constraint_violation > 0.0:
                pass
            elif self._calculate_fitness(solution, weights) < self._calculate_fitness(candidate, weights):
                replace = True
                
            if replace:
                self.population[i] = solution
                c = c + 1
                
            if c >= self.eta:
                break
    
    def initialize(self):
        self.population = []
        
        # initialize weights
        self.weights = random_weights(self.population_size, self.problem.nobjs)
        
        # initialize the neighborhoods based on weights
        self.neighborhoods = []
        
        for i in range(self.population_size):
            sorted_weights = [i[0] for i in sorted(enumerate(self.weights), key=lambda x:x[1])]
            self.neighborhoods.append(sorted_weights[:self.neighborhood_size])
            
        # initialize the ideal point
        self.ideal_point = [POSITIVE_INFINITY]*self.problem.nobjs
        
        # initialize the utilities and fitnesses
        self.utilities = [1.0]*self.population_size
        self.fitnesses = [0.0]*self.population_size
        
        # generate and evaluate the initial population
        self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]
        self.evaluateAll(self.population)
        
        # update the ideal point
        for i in range(self.population_size):
            self._update_ideal(self.population[i])
        
        # compute fitness
        for i in range(self.population_size):
            self.fitnesses[i] = self._calculate_fitness(self.population[i], self.weights[i])
            
    def _get_subproblems(self):
        """ Determines the subproblems to search.
        
        If :code:`utility_update` has been set, then this method follows the
        utility-based MOEA/D search.  Otherwise, it follows the original MOEA/D
        specification.
        """
        indices = []
        
        if self.update_utility is None:
            indices.extend(range(self.population_size))
        else:
            indices = range(self.problem.nobjs)
            
            for _ in range(self.problem.nobjs, self.population_size / 5):
                index = random.randrange(self.population_size)
                
                for _ in range(9):
                    temp_index = random.randrange(self.population_size)
                    
                    if self.utilities[temp_index] > self.utilities[index]:
                        index = temp_index
            
            indices.append(index)
            
        random.shuffle(indices)
        return indices
    
    def _get_mating_indices(self, index):
        """Determines the mating indices.
        
        Returns the population members that are considered during mating.  With
        probability :code:`delta`, the neighborhood is returned.  Otherwise,
        the entire population is returned.
        """
        if random.uniform(0.0, 1.0) <= self.delta:
            return self.neighborhoods[index]
        else:
            return range(self.population_size)
        
    def _update_utility(self):
        for i in range(self.population_size):
            old_fitness = self.fitnesses[i]
            new_fitness = self._calculate_fitness(self.population[i], self.weights[i])

            if old_fitness - new_fitness > 0.001:
                self.utilities[i] = 1.0
            else:
                self.utilities[i] = min(1.0, 0.95 * (1.0 + self.delta / 0.001) * self.utilities[i])
            
            self.fitnesses[i] = new_fitness
            
    def iterate(self):
        for index in self._get_subproblems():
            mating_indices = self._get_mating_indices(index)
            parents = [self.population[index]] + [self.population[i] for i in mating_indices[:(self.variator.arity-1)]]
            offspring = self.variator.evolve(parents)
            
            self.evaluateAll(offspring)
            
            for child in offspring:
                self._update_ideal(child)
                self._update_solution(child, mating_indices)
                
        self.generation += 1
        
        if self.update_utility >= 0 and self.generation % self.update_utility == 0:
            self._update_utility()