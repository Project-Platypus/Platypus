# Copyright 2015-2024 David Hadka
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
import random
import sys
import warnings
from abc import ABCMeta, abstractmethod

from .config import PlatypusConfig
from .core import Algorithm, nondominated_sort_cmp
from .operators import UM


def default_variator(problem):
    warnings.warn("default_variator(...) is being deprecated, please use PlatypusConfig.default_variator(...) instead",
                  DeprecationWarning, stacklevel=2)
    return PlatypusConfig.default_variator(problem)

def default_mutator(problem):
    warnings.warn("default_mutator(...) is being deprecated, please use PlatypusConfig.default_mutator(...) instead",
                  DeprecationWarning, stacklevel=2)
    return PlatypusConfig.default_variator(problem)

def nondominated_cmp(x, y):
    warnings.warn("nondominated_cmp(...) is being deprecated, please use nondominated_sort_cmp(...) instead",
                  DeprecationWarning, stacklevel=2)
    return nondominated_sort_cmp(x, y)

class PeriodicAction(Algorithm, metaclass=ABCMeta):

    def __init__(self,
                 algorithm,
                 frequency=10000,
                 by_nfe=True):
        super().__init__(algorithm.problem, algorithm.evaluator)
        self.algorithm = algorithm
        self.frequency = frequency
        self.by_nfe = by_nfe
        self.iteration = 0
        self.last_invocation = 0

        warnings.warn(f"{type(self).__name__} is being deprecated, please switch to using extensions",
                      DeprecationWarning, stacklevel=2)

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
        """Performs the action."""
        pass

    def __getattr__(self, name):
        # Be careful to not interfere with multiprocessing's unpickling, where it may check for
        # an attribute before the "algorithm" attribute is set.  Without this guard in place, we
        # would get stuck in an infinite loop looking for the "algorithm" attribute.
        if "algorithm" in self.__dict__:
            return getattr(self.algorithm, name)
        if sys.version_info >= (3, 10):
            raise AttributeError(name=name, obj=self)
        else:
            raise AttributeError()

class AdaptiveTimeContinuation(PeriodicAction):

    def __init__(self,
                 algorithm,
                 window_size=100,
                 max_window_size=1000,
                 population_ratio=4.0,
                 min_population_size=10,
                 max_population_size=10000,
                 mutator=UM(1.0)):
        super().__init__(algorithm,
                         frequency=window_size,
                         by_nfe=False)
        self.window_size = window_size
        self.max_window_size = max_window_size
        self.population_ratio = population_ratio
        self.min_population_size = min_population_size
        self.max_population_size = max_population_size
        self.mutator = mutator
        self.last_restart = 0

    def check(self):
        """Checks if a restart is required."""
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
        """Performs the restart procedure."""
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
                 window_size=100,
                 max_window_size=1000,
                 population_ratio=4.0,
                 min_population_size=10,
                 max_population_size=10000,
                 mutator=UM(1.0)):
        super().__init__(algorithm,
                         window_size,
                         max_window_size,
                         population_ratio,
                         min_population_size,
                         max_population_size,
                         mutator)
        self.last_improvements = 0

    def check(self):
        result = super().check()

        if not result:
            if self.archive.improvements <= self.last_improvements:
                result = True

        self.last_improvements = self.archive.improvements
        return result

    def restart(self):
        super().restart()
        self.last_improvements = self.archive.improvements
