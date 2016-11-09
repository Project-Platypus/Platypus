# Copyright 2015-2016 David Hadka
#
# This file is part of Platypus, a Python module for designing and using
# evolutionary algorithms (EAs) and multiobjective evolutionary algorithms
# (MOEAs).
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
from __future__ import absolute_import, division, print_function

import random
from collections import deque
from .core import Algorithm, EpsilonBoxArchive
from .operators import UM, RandomGenerator, TournamentSelector, Multimethod,\
    GAOperator, SBX, PM, UM, PCX, UNDX, SPX, DifferentialEvolution
from .algorithms import EpsMOEA, NSGAII
from abc import ABCMeta, abstractmethod

class PeriodicAction(Algorithm):
    
    __metaclass__ = ABCMeta
    
    def __init__(self,
                 algorithm,
                 frequency = 10000,
                 by_nfe = True):
        super(PeriodicAction, self).__init__(algorithm.problem,
                                             algorithm.evaluator)
        self.algorithm = algorithm
        self.frequency = frequency
        self.by_nfe = by_nfe
        self.iteration = 0
        self.last_invocation = 0
        
    def step(self):
        self.algorithm.step()
        self.iteration += 1
        self.nfe = self.algorithm.nfe
        
        if self.by_nfe:
            if self.nfe - self.last_invocation >= self.frequency:
                self.do_action()
                self.last_invocation = self.nfe
        else:
            if self.iteration - self.last_invocation >= self.frequency:
                self.do_action()
                self.last_invocation = self.iteration
                
    @abstractmethod
    def do_action(self):
        raise NotImplementedError("method not implemented")
        
    def __getattr__(self, name):
        return getattr(self.algorithm, name)
        
class AdaptiveTimeContinuation(PeriodicAction):
    
    def __init__(self,
                 algorithm,
                 window_size = 100,
                 max_window_size = 1000,
                 population_ratio = 4.0,
                 min_population_size = 10,
                 max_population_size = 10000,
                 mutator = UM(1.0)):
        super(AdaptiveTimeContinuation, self).__init__(algorithm,
                                                       frequency = window_size,
                                                       by_nfe = False)
        self.window_size = window_size
        self.max_window_size = max_window_size
        self.population_ratio = population_ratio
        self.min_population_size = min_population_size
        self.max_population_size = max_population_size
        self.mutator = mutator
        self.last_restart = 0
        
    def check(self):
        population_size = len(self.algorithm.population)
        target_size = self.population_ratio * len(self.algorithm.archive)
        
        if self.iteration - self.last_restart >= self.max_window_size:
            return True
        elif (target_size >= self.min_population_size and
              target_size <= self.max_population_size and
              abs(population_size - target_size) > (0.25 * target_size)):
            return True
        else:
            return False
        
    def restart(self):
        archive = self.algorithm.archive
        population = archive[:]
            
        new_size = int(self.population_ratio * len(archive))
        
        if new_size < self.min_population_size:
            new_size = self.min_population_size
        elif new_size > self.max_population_size:
            new_size = self.max_population_size
            
        offspring = []
            
        while len(population) + len(offspring) < new_size:
            parents = [archive[random.randrange(len(archive))] for _ in range(self.mutator.arity)]
            offspring.extend(self.mutator.evolve(parents))
            
        self.algorithm.evaluate_all(offspring)
        self.nfe = self.algorithm.nfe
        
        population.extend(offspring)
        archive.extend(offspring)
        
        self.last_restart = self.iteration  
        self.algorithm.population = population
        self.algorithm.population_size = len(population)
        
    def do_action(self):
        if self.check():
            self.restart()
            
class EpsilonProgressContinuation(AdaptiveTimeContinuation):
    
    def __init__(self,
                 algorithm,
                 window_size = 100,
                 max_window_size = 1000,
                 population_ratio = 4.0,
                 min_population_size = 10,
                 max_population_size = 10000,
                 mutator = UM(1.0)):
        super(EpsilonProgressContinuation, self).__init__(algorithm,
                                                          window_size,
                                                          max_window_size,
                                                          population_ratio,
                                                          min_population_size,
                                                          max_population_size,
                                                          mutator)
        self.last_improvements = 0
        
    def check(self):
        result = super(EpsilonProgressContinuation, self).check()
        
        if not result:
            if self.archive.improvements <= self.last_improvements:
                result = True
            
        self.last_improvements = self.archive.improvements
        return result
    
    def restart(self):
        super(EpsilonProgressContinuation, self).restart()
        self.last_improvements = self.archive.improvements
        
class EpsNSGAII(AdaptiveTimeContinuation):
    
    def __init__(self,
                 problem,
                 epsilons,
                 population_size = 100,
                 generator = RandomGenerator(),
                 selector = TournamentSelector(2),
                 variator = None,
                 **kwargs):
        super(EpsNSGAII, self).__init__(
                NSGAII(problem,
                       population_size,
                       generator,
                       selector,
                       variator,
                       EpsilonBoxArchive(epsilons),
                       **kwargs))
        
class BorgMOEA(EpsilonProgressContinuation):
    
    def __init__(self, problem,
                 epsilons,
                 population_size = 100,
                 generator = RandomGenerator(),
                 selector = TournamentSelector(2),
                 recency_list_size = 50,
                 max_mutation_index = 10,
                 **kwargs):
        super(BorgMOEA, self).__init__(
                EpsMOEA(problem,
                        epsilons,
                        population_size,
                        generator,
                        selector,
                        **kwargs))
        self.recency_list = deque()
        self.recency_list_size = recency_list_size
        self.restarted_last_check = False
        self.base_mutation_index = 0
        self.max_mutation_index = max_mutation_index
        
        # overload the variator and iterate method
        self.algorithm.variator = Multimethod(self, [
                GAOperator(SBX(), PM()),
                DifferentialEvolution(),
                UM(),
                PCX(),
                UNDX(),
                SPX()])
        
        self.algorithm.iterate = self.iterate
        
    def do_action(self):
        if self.check():
            if self.restarted_last_check:
                self.base_mutation_index = min(self.base_mutation_index+1,
                                               self.max_mutation_index)
                 
            # update the mutation probability prior to restart
            probability = self.base_mutation_index / float(self.max_mutation_index)
            probability = probability + (1.0 - probability)/self.algorithm.problem.nvars
            self.mutator.probability = probability
             
            self.restart()
            self.restarted_last_check = True
        else:
            if self.restarted_last_check:
                self.base_mutation_index = max(self.base_mutation_index-1, 0)
             
            self.restarted_last_check = False
        
    def iterate(self):
        if len(self.archive) <= 1:
            parents = self.selector.select(self.variator.arity, self.population)
        else:
            parents = self.selector.select(self.variator.arity-1, self.population) + [random.choice(self.archive)]
 
        random.shuffle(parents)
         
        children = self.variator.evolve(parents)
         
        self.algorithm.evaluate_all(children)
        self.nfe = self.algorithm.nfe
         
        for child in children:
            self._add_to_population(child)
             
            if self.archive.add(child):
                self.recency_list.append(child)
                  
                if len(self.recency_list) > self.recency_list_size:
                    self.recency_list.popleft()