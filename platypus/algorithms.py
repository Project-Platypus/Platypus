import sys
import math
import random
import operator
import itertools
from sets import Set
from abc import ABCMeta, abstractmethod
from platypus.core import Algorithm, Variator, Dominance, ParetoDominance,\
    AttributeDominance, nondominated_sort, nondominated_prune,\
    nondominated_truncate, nondominated_split,\
    EPSILON, POSITIVE_INFINITY
from platypus.operators import TournamentSelector, RandomGenerator, DifferentialEvolution
from platypus.tools import DistanceMatrix
from platypus.weights import random_weights, chebyshev, normal_boundary_weights
            
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
            indices = []
            
            if self.weight_generator == random_weights:
                indices.extend(range(self.problem.nobjs))
            
            while len(indices) < self.population_size:
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

def _find_extreme_points(solutions, objective):
    eps = 0.000001
    nobjs = solutions[0].problem.nobjs
    
    weights = [eps]*nobjs
    weights[objective] = 1.0
    
    min_index = -1
    min_value = POSITIVE_INFINITY
    
    for i in range(len(solutions)):
        objectives = solutions[i].normalized_objectives
        value = max([objectives[i]/weights[i] for i in range(nobjs)])
        
        if value < min_value:
            min_index = i
            min_value = value
            
    return solutions[min_index]

def _point_line_distance(line, point):
    n = len(line)
    lp_dot = reduce(operator.add, [line[i]*point[i] for i in range(n)], 0)
    ll_dot = reduce(operator.add, [line[i]*line[i] for i in range(n)], 0)
    pline = [(lp_dot / ll_dot) * line[i] for i in range(n)]
    diff = [pline[i] - point[i] for i in range(n)]
    return math.sqrt(reduce(operator.add, [diff[i]*diff[i] for i in range(n)], 0))

def _associate_to_reference_point(solutions, reference_points):
    result = [[] for _ in range(len(reference_points))]
    
    for solution in solutions:
        min_index = -1
        min_distance = POSITIVE_INFINITY
        
        for i in range(len(reference_points)):
            distance = _point_line_distance(reference_points[i], solution.normalized_objectives)
    
            if distance < min_distance:
                min_index = i
                min_distance = distance
                
        result[min_index].append(solution)
        
    return result

def _find_minimum_distance(solutions, reference_point):
    min_index = -1
    min_distance = POSITIVE_INFINITY
        
    for i in range(len(solutions)):
        solution = solutions[i]
        distance = _point_line_distance(reference_point, solution.normalized_objectives)
    
        if distance < min_distance:
            min_index = i
            min_distance = distance
                
    return solutions[min_index]
    
def reference_point_truncate(solutions, size, ideal_point, reference_points):
    from numpy.linalg import lstsq, LinAlgError
    nobjs = solutions[0].problem.nobjs
    
    if len(solutions) > size:
        result, remaining = nondominated_split(solutions, size)
        
        # update the ideal point
        for solution in solutions:
            for i in range(nobjs):
                ideal_point[i] = min(ideal_point[i], solution.objectives[i])
                
        # translate points by ideal point
        for solution in solutions:
            solution.normalized_objectives = [solution.objectives[i] - ideal_point[i] for i in range(nobjs)]
        
        # find the extreme points
        extreme_points = [_find_extreme_points(solutions, i) for i in range(nobjs)]
        
        # calculate the intercepts
        degenerate = False
        
        try:
            b = [1.0]*nobjs
            A = [s.normalized_objectives for s in extreme_points]
            x = lstsq(A, b)
            intercepts = [1.0 / i for i in x]
        except LinAlgError:
            degenerate = True
            
        if not degenerate:
            for i in range(nobjs):
                if intercepts[i] < 0.001:
                    degenerate = True
                    break
                
        if degenerate:
            intercepts = [-POSITIVE_INFINITY]*nobjs
            
            for i in range(nobjs):
                intercepts[i] = max([s.normalized_objectives for s in solutions] + [EPSILON])

        # normalize objectives using intercepts
        for solution in solutions:
            solution.normalized_objectives = [solution.normalized_objectives[i] / intercepts[i] for i in range(nobjs)]

        # associate each solution to a reference point
        members = _associate_to_reference_point(result, reference_points)
        potential_members = _associate_to_reference_point(remaining, reference_points)
        excluded = Set()
        
        while len(result) < size:
            # identify reference point with the fewest associated members
            min_indices = []
            min_count = sys.maxint
            
            for i in range(len(members)):
                if i not in excluded and len(members[i]) <= min_count:
                    if len(members[i]) < min_count:
                        min_indices = []
                        min_count = len(members[i])
                    min_indices.append(i)
            
            # pick one randomly if there are multiple options
            min_index = random.choice(min_indices)
            
            # add associated solution
            if min_count == 0:
                if len(potential_members[min_index]) == 0:
                    excluded.add(min_index)
                else:
                    min_solution = _find_minimum_distance(potential_members[min_index], reference_points[min_index])
                    result.append(min_solution)
                    members[min_index].append(min_solution)
                    potential_members[min_index].remove(min_solution)
            else:
                if len(potential_members[min_index]) == 0:
                    excluded.add(min_index)
                else:
                    rand_solution = random.choice(potential_members[min_index])
                    result.append(rand_solution)
                    members[min_index].append(rand_solution)
                    potential_members[min_index].remove(rand_solution)
                    
        return result
    else:
        return solutions

class NSGAIII(GeneticAlgorithm):
    
    def __init__(self, problem,
                 divisions_outer,
                 divisions_inner = 0,
                 population_size = 100,
                 generator = RandomGenerator(),
                 selector = TournamentSelector(2),
                 variator = None):
        super(NSGAIII, self).__init__(problem, generator)
        self.population_size = population_size
        self.selector = selector
        self.variator = variator
        self.ideal_point = [POSITIVE_INFINITY]*problem.nobjs
        self.reference_points = normal_boundary_weights(problem.nobjs, divisions_outer, divisions_inner)
        
    def iterate(self):
        offspring = []
        
        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))
            
        self.evaluateAll(offspring)
        
        offspring.extend(self.population)
        nondominated_sort(offspring)
        self.population = reference_point_truncate(offspring, self.population_size, self.ideal_point, self.reference_points)
