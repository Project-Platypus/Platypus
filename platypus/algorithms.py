# Copyright 2015 David Hadka
#
# This file is part of Platypus.
#
# Platypus is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Platypus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Platypus.  If not, see <http://www.gnu.org/licenses/>.
import sys
import copy
import math
import random
import operator
import itertools
from sets import Set
from abc import ABCMeta, abstractmethod
from platypus.core import Algorithm, Variator, Dominance, ParetoDominance, AttributeDominance,\
    AttributeDominance, nondominated, nondominated_sort, nondominated_prune,\
    nondominated_truncate, nondominated_split, crowding_distance,\
    EPSILON, POSITIVE_INFINITY, truncate_fitness, Archive, EpsilonDominance, \
    FitnessArchive, Solution, HypervolumeFitnessEvaluator
from platypus.operators import TournamentSelector, RandomGenerator, DifferentialEvolution,\
    clip, Mutation, UniformMutation, NonUniformMutation
from platypus.tools import DistanceMatrix, choose, point_line_dist, lsolve,\
    tred2, tql2, check_eigensystem
from platypus.weights import random_weights, chebyshev, normal_boundary_weights
from __builtin__ import True
            
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

class EpsilonMOEA(GeneticAlgorithm):
    
    def __init__(self, problem,
                 epsilons,
                 population_size = 100,
                 generator = RandomGenerator(),
                 selector = TournamentSelector(2),
                 variator = None):
        super(EpsilonMOEA, self).__init__(problem, generator)
        self.population_size = population_size
        self.selector = selector
        self.variator = variator
        self.dominance = ParetoDominance()
        self.archive = Archive(EpsilonDominance(epsilons))
        
    def step(self):
        if self.nfe == 0:
            self.initialize()
            self.result = self.archive
        else:
            self.iterate()
            self.result = self.archive
        
    def initialize(self):
        super(EpsilonMOEA, self).initialize()
        self.archive += self.population
        
    def iterate(self):
        if len(self.archive) <= 1:
            parents = self.selector.select(self.variator.arity, self.population)
        else:
            parents = self.selector.select(self.variator.arity-1, self.population) + [random.choice(self.archive)]

        random.shuffle(parents)
        
        children = self.variator.evolve(parents)
        self.evaluateAll(children)
        
        for child in children:
            self._add_to_population(child)
            self.archive.add(child)
            
    def _add_to_population(self, solution):
        dominates = []
        dominated = False
        
        for i in range(self.population_size):
            flag = self.dominance.compare(solution, self.population[i])

            if flag < 0:
                dominates.append(i)
            elif flag > 0:
                dominated = True
                
        if len(dominates) > 0:
            del self.population[random.choice(dominates)]
            self.population.append(solution)
        elif not dominated:
            self.population.remove(random.choice(self.population))
            self.population.append(solution)

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
            
    def _sort_weights(self, base, weights):
        """Returns the index of weights nearest to the base weight."""
        def compare(weight1, weight2):
            dist1 = math.sqrt(sum([math.pow(base[i]-weight1[1][i], 2.0) for i in range(len(base))]))
            dist2 = math.sqrt(sum([math.pow(base[i]-weight2[1][i], 2.0) for i in range(len(base))]))
            return cmp(dist1, dist2)
        
        sorted_weights = sorted(enumerate(weights), cmp=compare)
        return [i[0] for i in sorted_weights]
    
    def initialize(self):
        self.population = []
        
        # initialize weights
        self.weights = random_weights(self.population_size, self.problem.nobjs)
        
        # initialize the neighborhoods based on weights
        self.neighborhoods = []
        
        for i in range(self.population_size):
            sorted_weights = self._sort_weights(self.weights[i], self.weights)
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

class NSGAIII(GeneticAlgorithm):
    
    def __init__(self, problem,
                 divisions_outer,
                 divisions_inner = 0,
                 generator = RandomGenerator(),
                 selector = TournamentSelector(2),
                 variator = None):
        super(NSGAIII, self).__init__(problem, generator)
        self.selector = selector
        self.variator = variator
        
        self.population_size = choose(problem.nobjs + divisions_outer - 1, divisions_outer) + \
                (0 if divisions_inner == 0 else choose(problem.nobjs + divisions_inner - 1, divisions_inner))
        self.population_size = int(math.ceil(self.population_size / 4.0)) * 4;

        self.ideal_point = [POSITIVE_INFINITY]*problem.nobjs
        self.reference_points = normal_boundary_weights(problem.nobjs, divisions_outer, divisions_inner)
    
    def _find_extreme_points(self, solutions, objective):
        nobjs = self.problem.nobjs
        
        weights = [0.000001]*nobjs
        weights[objective] = 1.0
        
        min_index = -1
        min_value = POSITIVE_INFINITY
        
        for i in range(len(solutions)):
            objectives = solutions[i].normalized_objectives
            value = max([objectives[j]/weights[j] for j in range(nobjs)])
            
            if value < min_value:
                min_index = i
                min_value = value
                
        return solutions[min_index]

    def _associate_to_reference_point(self, solutions, reference_points):
        result = [[] for _ in range(len(reference_points))]
        
        for solution in solutions:
            min_index = -1
            min_distance = POSITIVE_INFINITY
            
            for i in range(len(reference_points)):
                distance = point_line_dist(solution.normalized_objectives, reference_points[i])
        
                if distance < min_distance:
                    min_index = i
                    min_distance = distance
                    
            result[min_index].append(solution)
            
        return result
    
    def _find_minimum_distance(self, solutions, reference_point):
        min_index = -1
        min_distance = POSITIVE_INFINITY
            
        for i in range(len(solutions)):
            solution = solutions[i]
            distance = point_line_dist(solution.normalized_objectives, reference_point)
        
            if distance < min_distance:
                min_index = i
                min_distance = distance
                    
        return solutions[min_index]
        
    def _reference_point_truncate(self, solutions, size):
        nobjs = self.problem.nobjs
        
        if len(solutions) > size:
            result, remaining = nondominated_split(solutions, size)

            # update the ideal point
            for solution in solutions:
                for i in range(nobjs):
                    self.ideal_point[i] = min(self.ideal_point[i], solution.objectives[i])
                    
            # translate points by ideal point
            for solution in solutions:
                solution.normalized_objectives = [solution.objectives[i] - self.ideal_point[i] for i in range(nobjs)]
            
            # find the extreme points
            extreme_points = [self._find_extreme_points(solutions, i) for i in range(nobjs)]
            
            # calculate the intercepts
            degenerate = False
            
            try:
                b = [1.0]*nobjs
                A = [s.normalized_objectives for s in extreme_points]
                x = lsolve(A, b)
                intercepts = [1.0 / i for i in x]
            except:
                degenerate = True
                
            if not degenerate:
                for i in range(nobjs):
                    if intercepts[i] < 0.001:
                        degenerate = True
                        break
                    
            if degenerate:
                intercepts = [-POSITIVE_INFINITY]*nobjs
                
                for i in range(nobjs):
                    intercepts[i] = max([s.normalized_objectives[i] for s in solutions] + [EPSILON])
    
            # normalize objectives using intercepts
            for solution in solutions:
                solution.normalized_objectives = [solution.normalized_objectives[i] / intercepts[i] for i in range(nobjs)]
    
            # associate each solution to a reference point
            members = self._associate_to_reference_point(result, self.reference_points)
            potential_members = self._associate_to_reference_point(remaining, self.reference_points)
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
                        min_solution = self._find_minimum_distance(potential_members[min_index], self.reference_points[min_index])
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
    
    def iterate(self):
        offspring = []
        
        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))
            
        self.evaluateAll(offspring)
        
        offspring.extend(self.population)
        nondominated_sort(offspring)
        self.population = self._reference_point_truncate(offspring, self.population_size)

class ParticleSwarm(Algorithm):
    
    def __init__(self, problem,
                 swarm_size = 100,
                 leader_size = 100,
                 generator = RandomGenerator(),
                 mutate = None,
                 leader_comparator = AttributeDominance("fitness"),
                 dominance = ParetoDominance(),
                 fitness = None,
                 larger_preferred = True,
                 fitness_getter = operator.attrgetter("fitness")):
        super(ParticleSwarm, self).__init__(problem)
        self.swarm_size = swarm_size
        self.leader_size = leader_size
        self.generator = generator
        self.mutate = mutate
        self.leader_comparator = leader_comparator
        self.dominance = dominance
        self.fitness = fitness
        self.larger_preferred = larger_preferred
        self.fitness_getter = fitness_getter
        
    def step(self):
        if self.nfe == 0:
            self.initialize()
            self.result = self.leaders
        else:
            self.iterate()
            self.result = self.leaders
            
    def initialize(self):
        self.particles = [self.generator.generate(self.problem) for _ in range(self.swarm_size)]
        self.evaluateAll(self.particles)
        
        self.local_best = self.particles[:]
        
        self.leaders = FitnessArchive(self.fitness,
                                      larger_preferred = self.larger_preferred,
                                      getter = self.fitness_getter)
        self.leaders += self.particles
        self.leaders.truncate(self.leader_size)
        
        self.velocities = [[0.0]*self.problem.nvars for _ in range(self.swarm_size)]
    
    def iterate(self):
        self._update_velocities()
        self._update_positions()
        self._mutate()
        self.evaluateAll(self.particles)
        self._update_local_best()
        
        self.leaders += self.particles
        self.leaders.truncate(self.leader_size)
    
    def _update_velocities(self):
        for i in range(self.swarm_size):
            particle = self.particles[i].variables
            local_best = self.local_best[i].variables
            leader = self._select_leader().variables
            
            r1 = random.uniform(0.0, 1.0)
            r2 = random.uniform(0.0, 1.0)
            C1 = random.uniform(1.5, 2.0)
            C2 = random.uniform(1.5, 2.0)
            W = random.uniform(0.1, 0.5)
            
            for j in range(self.problem.nvars):
                self.velocities[i][j] = W * self.velocities[i][j] + \
                        C1*r1*(local_best[j] - particle[j]) + \
                        C2*r2*(leader[j] - particle[j])
    
    def _select_leader(self):
        leader1 = random.choice(self.leaders)
        leader2 = random.choice(self.leaders)
        flag = self.leader_comparator.compare(leader1, leader2)
        
        if flag < 0:
            return leader1
        elif flag > 0:
            return leader2
        elif bool(random.getrandbits(1)):
            return leader1
        else:
            return leader2
    
    def _update_positions(self):
        for i in range(self.swarm_size):
            offspring = copy.deepcopy(self.particles[i])
            
            for j in range(self.problem.nvars):
                type = self.problem.types[j]
                value = offspring.variables[j] + self.velocities[i][j]
                
                if value < type.min_value:
                    value = type.min_value
                    self.velocities[i][j] *= -1
                elif value > type.max_value:
                    value = type.max_value
                    self.velocities[i][j] *= -1
                    
                offspring.variables[j] = value
                
            offspring.evaluated = False
            self.particles[i] = offspring
    
    def _update_local_best(self):
        for i in range(self.swarm_size):
            flag = self.dominance.compare(self.particles[i], self.local_best[i])
            
            if flag <= 0:
                self.local_best[i] = self.particles[i]
                
    def _mutate(self):
        if self.mutate is not None:
            for i in range(self.swarm_size):
                self.particles[i] = self.mutate.mutate([self.particles[i]])[0]
                
class OMOPSO(ParticleSwarm):
     
    def __init__(self, problem,
                 epsilons,
                 swarm_size = 100,
                 leader_size = 100,
                 generator = RandomGenerator(),
                 mutation_probability = 0.1,
                 mutation_perturbation = 0.5,
                 max_iterations = 100):
        super(OMOPSO, self).__init__(problem,
                                     swarm_size=swarm_size,
                                     leader_size=leader_size,
                                     generator = generator,
                                     leader_comparator = AttributeDominance("crowding_distance"),
                                     dominance = ParetoDominance(),
                                     fitness = crowding_distance,
                                     fitness_getter = operator.attrgetter("crowding_distance"))
        self.max_iterations = max_iterations
        self.archive = Archive(EpsilonDominance(epsilons))
        self.uniform_mutation = UniformMutation(mutation_probability,
                                                mutation_perturbation)
        self.nonuniform_mutation = NonUniformMutation(mutation_probability,
                                                      mutation_perturbation,
                                                      max_iterations,
                                                      self)
        
    def step(self):
        if self.nfe == 0:
            self.initialize()
            self.result = self.archive
        else:
            self.iterate()
            self.result = self.archive
    
    def initialize(self):
        super(OMOPSO, self).initialize()
        self.archive += self.particles
        
    def iterate(self):
        super(OMOPSO, self).iterate()
        self.archive += self.particles
        
    def _mutate(self):
        for i in range(self.swarm_size):
            if i % 3 == 0:
                self.particles[i] = self.nonuniform_mutation.mutate(self.particles[i])
            elif i % 3 == 1:
                self.particles[i] = self.uniform_mutation.mutate(self.particles[i])

class SMPSO(ParticleSwarm):
     
    def __init__(self, problem,
                 swarm_size = 100,
                 leader_size = 100,
                 generator = RandomGenerator(),
                 mutation_probability = 0.1,
                 mutation_perturbation = 0.5,
                 max_iterations = 100,
                 mutate = None):
        super(SMPSO, self).__init__(problem,
                                    swarm_size=swarm_size,
                                    leader_size=leader_size,
                                    generator = generator,
                                    leader_comparator = AttributeDominance("crowding_distance"),
                                    dominance = ParetoDominance(),
                                    fitness = crowding_distance,
                                    fitness_getter = operator.attrgetter("crowding_distance"),
                                    mutate = mutate)
        self.max_iterations = max_iterations
        self.maximum_velocity = [(t.max_value - t.min_value)/2.0 for t in problem.types]
        self.minimum_velocity = [-(t.max_value - t.min_value)/2.0 for t in problem.types]
    
    def _update_velocities(self):
        for i in range(self.swarm_size):
            particle = self.particles[i].variables
            local_best = self.local_best[i].variables
            leader = self._select_leader().variables
            
            r1 = random.uniform(0.0, 1.0)
            r2 = random.uniform(0.0, 1.0)
            C1 = random.uniform(1.5, 2.5)
            C2 = random.uniform(1.5, 2.5)
            W = random.uniform(0.1, 0.1)
            
            for j in range(self.problem.nvars):
                self.velocities[i][j] = self._constriction(C1, C2) * \
                        (W * self.velocities[i][j] + \
                        C1*r1*(local_best[j] - particle[j]) + \
                        C2*r2*(leader[j] - particle[j]))
                        
                self.velocities[i][j] = clip(self.velocities[i][j],
                                             self.minimum_velocity[j],
                                             self.maximum_velocity[j])

    def _constriction(self, C1, C2):
        rho = C1 + C2
        
        if rho <= 4:
            return 1.0
        else:
            return 2.0 / (2.0 - rho - math.sqrt(math.pow(rho, 2.0) - 4.0*rho))
        
    def _mutate(self):
        for i in range(self.swarm_size):
            if i % 6 == 0:
                self.particles[i] = self.mutate.mutate(self.particles[i])
                
class CMAES(Algorithm):
    
    def __init__(self, problem,
                 offspring_size = 100,
                 cc = None,
                 cs = None,
                 damps = None,
                 ccov = None,
                 ccovsep = None,
                 sigma = None,
                 diagonal_iterations = None,
                 indicator = "crowding",
                 initial_search_point = None,
                 check_consistency = False,
                 epsilons = None,
                 fitness = None):
        super(CMAES, self).__init__(problem)
        self.offspring_size = offspring_size
        self.cc = cc
        self.cs = cs
        self.damps = damps
        self.ccov = ccov
        self.ccovsep = ccovsep
        self.sigma = sigma
        self.diagonal_iterations = diagonal_iterations
        self.indicator = indicator
        self.initial_search_point = initial_search_point
        self.check_consistency = check_consistency
        self.fitness = fitness
        self.population = []
        self.iteration = 0
        self.last_eigenupdate = 0
        
        if epsilons is None:
            self.archive = Archive()
        else:
            self.archive = Archive(EpsilonDominance(epsilons))
        
    def step(self):
        if self.nfe == 0:
            self.initialize()
            self.iterate()
            self.result = self.population
        else:
            self.iterate()
            self.result = self.population
            print self.nfe
        
    def initialize(self):
        if self.sigma is None:
            self.sigma = 0.5
            
        if self.diagonal_iterations is None:
            self.diagonal_iterations = 150 * self.problem.nvars / self.offspring_size
            
        self.diag_D = [1.0]*self.problem.nvars
        self.pc = [0.0]*self.problem.nvars
        self.ps = [0.0]*self.problem.nvars
        self.B = [[1.0 if i==j else 0.0 for j in range(self.problem.nvars)] for i in range(self.problem.nvars)]
        self.C = [[1.0 if i==j else 0.0 for j in range(self.problem.nvars)] for i in range(self.problem.nvars)]
        self.xmean = [0.0]*self.problem.nvars
        
        if self.initial_search_point is None:
            for i in range(self.problem.nvars):
                type = self.problem.types[i]
                offset = self.sigma * self.diag_D[i]
                rangev = type.max_value - type.min_value - 2*self.sigma*self.diag_D[i]
                
                if offset > 0.4 * (type.max_value - type.min_value):
                    offset = 0.4 * (type.max_value - type.min_value)
                    rangev = 0.2 * (type.max_value - type.min_value)
                    
                self.xmean[i] = type.min_value + offset + random.uniform(0.0, 1.0) * rangev
        else:
            for i in range(self.problem.nvars):
                self.xmean[i] = self.initial_search_point[i] + self.sigma*self.diag_D[i]*random.gauss()  
            
        self.chi_N = math.sqrt(self.problem.nvars) * (1.0 - 1.0/(4.0*self.problem.nvars) + 1.0/(21.0*self.problem.nvars**2))
        self.mu = int(math.floor(self.offspring_size / 2.0))
        self.weights = [math.log(self.mu + 1.0) - math.log(i + 1.0) for i in range(self.mu)]
        
        sum_of_weights = sum(self.weights)
        self.weights = [w / sum_of_weights for w in self.weights]
        
        sumsq_of_weights = sum([w**2 for w in self.weights])
        self.mueff = 1.0 / sumsq_of_weights
        
        if self.cs is None:
            self.cs = (self.mueff + 2.0) / (self.problem.nvars + self.mueff + 3.0)
            
        if self.damps is None:
            self.damps = (1.0 + 2.0*max(0, math.sqrt((self.mueff - 1.0) / (self.problem.nvars + 1.0)) - 1.0)) + self.cs
            
        if self.cc is None:
            self.cc = 4.0 / (self.problem.nvars + 4.0)
            
        if self.ccov is None:
            self.ccov = 2.0 / (self.problem.nvars + 1.41) / (self.problem.nvars + 1.41) / self.mueff + (1.0 - (1.0 / self.mueff)) * min(1.0, (2.0*self.mueff - 1.0) / (self.mueff + (self.problem.nvars + 2.0)**2))
            
        if self.ccovsep is None:
            self.ccovsep = min(1.0, self.ccov * (self.problem.nvars + 1.5) / 3.0)
            
    def eigendecomposition(self):
        self.last_eigenupdate = self.iteration
        
        if self.diagonal_iterations >= self.iteration:
            for i in range(self.problem.nvars):
                self.diag_D[i] = math.sqrt(self.C[i][i])
        else:
            for i in range(self.problem.nvars):
                for j in range(i+1):
                    self.B[i][j] = self.B[j][i] = self.C[i][j]
                    
            offdiag = [0.0]*self.problem.nvars
            tred2(self.problem.nvars, self.B, self.diag_D, offdiag)
            tql2(self.problem.nvars, self.diag_D, offdiag, self.B)
            
            if self.check_consistency:
                check_eigensystem(self.problem.nvars, self.C, self.diag_D, self.B)
                
            for i in range(self.problem.nvars):
                if self.diag_D[i] < 0.0:
                    print >> sys.stderr, "an eigenvalue has become negative"
                    self.diag_D[i] = 0.0
                    
                self.diag_D[i] = math.sqrt(self.diag_D[i])
            
    def test_and_correct_numerics(self):
        if len(self.population) > 0:
            pass
            
    def sample(self):
        if (self.iteration - self.last_eigenupdate) > 1.0 / self.ccov / self.problem.nvars / 5.0:
            self.eigendecomposition()
            
        if self.check_consistency:
            self.test_and_correct_numerics()
            
        samples = []
        
        for i in range(self.offspring_size):
            solution = Solution(self.problem)
            
            if self.diagonal_iterations >= self.iteration:
                while True:
                    feasible = True
                    
                    for j in range(self.problem.nvars):
                        type = self.problem.types[j]
                        value = self.xmean[j] + self.sigma * self.diag_D[j] * random.gauss(0.0, 1.0)
                        if value < type.min_value or value > type.max_value:
                            feasible = False
                            break
                        
                        solution.variables[j] = value
                        
                    if feasible:
                        break
            else:
                artmp = [0.0]*self.problem.nvars
                
                while True:
                    feasible = True
                    
                    for j in range(self.problem.nvars):
                        artmp[j] = self.diag_D[j] * random.gauss(0.0, 1.0)
                        
                    for j in range(self.problem.nvars):
                        type = self.problem.types[j]
                        mutation = 0.0
                        
                        for k in range(self.problem.nvars):
                            mutation += self.B[j][k] * artmp[k]
                            
                        value = self.xmean[j] + self.sigma * mutation
                        
                        if value < type.min_value or value > type.max_value:
                            feasible = False
                            break
                        
                        solution.variables[j] = value
                        
                    if feasible:
                        break
                    
            samples.append(solution)
        
        self.iteration += 1
        return samples
    
    def update_distribution(self):
        xold = self.xmean[:]
        BDz = [0.0]*self.problem.nvars
        artmp = [0.0]*self.problem.nvars
        
        if self.problem.nobjs == 1:
            self.population = sorted(self.population, key=lambda x : x.objectives[0])
        else:
            if self.fitness is None:
                def comparator(x, y):
                    if x.rank == y.rank:
                        return cmp(-x.crowding_distance, -y.crowding_distance)
                    else:
                        return cmp(x.rank, y.rank)
    
                self.population = sorted(self.population, cmp=comparator) 
            else:
                self.population = sorted(self.population, cmp=AttributeDominance().compare)
            
        for i in range(self.problem.nvars):
            self.xmean[i] = 0.0
            
            for j in range(self.mu):
                self.xmean[i] += self.weights[j] * self.population[j].variables[i]
                      
            BDz[i] = math.sqrt(self.mueff) * (self.xmean[i] - xold[i]) / self.sigma
              
        if self.diagonal_iterations >= self.iteration:
            for i in range(self.problem.nvars):
                self.ps[i] = (1.0 - self.cs) * self.ps[i] + math.sqrt(self.cs * (2.0 - self.cs)) * BDz[i] / self.diag_D[i]
        else:
            for i in range(self.problem.nvars):
                artmp[i] = sum([self.B[j][i]*BDz[j] for j in range(self.problem.nvars)]) / self.diag_D[i]
            
            for i in range(self.problem.nvars):
                self.ps[i] = (1.0 - self.cs) * self.ps[i] + math.sqrt(self.cs * (2.0 - self.cs)) * sum([self.B[i][j] * artmp[j] for j in range(self.problem.nvars)])
              
        psxps = sum([self.ps[i]**2 for i in range(self.problem.nvars)])
        hsig = 0.0
        
        if math.sqrt(psxps) / math.sqrt(1.0 - math.pow(1.0 - self.cs, 2.0 * self.iteration)) / self.chi_N < 1.4 + 2.0 / (self.problem.nvars+1):
            hsig = 1.0
            
        for i in range(self.problem.nvars):
            self.pc[i] = (1.0 - self.cc) * self.pc[i] + hsig * math.sqrt(self.cc * (2.0 - self.cc)) * BDz[i]
            
        for i in range(self.problem.nvars):
            for j in range(i if self.diagonal_iterations >= self.iteration else 0, i+1):
                self.C[i][j] = (1.0 - (self.ccovsep if self.diagonal_iterations >= self.iteration else self.ccov)) * self.C[i][j] + self.ccov * (1.0 / self.mueff) * (self.pc[i] * self.pc[j] + (1.0 - hsig) * self.cc * (2.0 - self.cc) * self.C[i][j])
                
                for k in range(self.mu):
                    self.C[i][j] += self.ccov * (1.0 - 1.0 / self.mueff) * self.weights[k] * (self.population[k].variables[i] - xold[i]) * (self.population[k].variables[j] - xold[j]) / (self.sigma**2)
              
        self.sigma *= math.exp(((math.sqrt(psxps) / self.chi_N) - 1.0) * self.cs / self.damps)
        
    def iterate(self):
        self.population = self.sample()
        self.evaluateAll(self.population)
        
        if self.problem.nobjs > 1:
            nondominated_sort(self.population)
            
            if self.fitness is not None:
                self.fitness(self.population)

        self.archive += self.population
        self.update_distribution()
        
class IBEA(GeneticAlgorithm):
    
    def __init__(self, problem,
                 population_size = 100,
                 fitness_evaluator = HypervolumeFitnessEvaluator(),
                 fitness_comparator = AttributeDominance("fitness"),
                 variator = None):
        super(IBEA, self).__init__(problem)
        self.population_size = population_size
        self.fitness_evaluator = fitness_evaluator
        self.fitness_comparator = fitness_comparator
        self.selector = TournamentSelector(2, fitness_comparator)
        self.variator = variator
        
    def initialize(self):
        super(IBEA, self).initialize()
        self.fitness_evaluator.evaluate(self.population)
        
    def iterate(self):
        print self.nfe
        offspring = []
        
        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))
            
        self.evaluateAll(offspring)
        
        self.population.extend(offspring)
        self.fitness_evaluator.evaluate(self.population)
        
        while len(self.population) > self.population_size:
            self.fitness_evaluator.remove(self.population, self._find_worst())
        
    def _find_worst(self):
        index = 0
        
        for i in range(1, len(self.population)):
            if self.fitness_comparator.compare(self.population[index], self.population[i]) < 0:
                index = i
                
        return index
        