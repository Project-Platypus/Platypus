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
import datetime
import logging
import random
import time
from abc import ABCMeta, abstractmethod

from .io import save_json
from .operators import UM

LOGGER = logging.getLogger("Platypus")

class Extension:
    """Extends the functionality of an algorithm."""

    def start_run(self, algorithm):
        """Executes at the start of a run.

        This can be used to capture the initial state of a run.  However,
        note that the :code:`run` method can be called multiple times.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm being run.
        """
        pass

    def end_run(self, algorithm):
        """Executes at the end of a run.

        This can be used to perform any finalization or post-processing.
        However, note that the :code:`run` method can be called multiple
        times.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm being run.
        """
        pass

    def pre_step(self, algorithm):
        """Executes before the :code:`step` method.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm being run.
        """
        pass

    def post_step(self, algorithm):
        """Executes after the :code:`step` method.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm being run.
        """
        pass

class FixedFrequencyExtension(Extension, metaclass=ABCMeta):
    """Extension that performs an action at a fixed frequency.

    Parameters
    ----------
    frequency : int
        The frequency the action occurs.
    by_nfe : bool
        If :code:`True`, the frequency is given in number of function
        evaluations.  If :code:`False`, the frequency is given in the number
        of iterations.
    """

    def __init__(self, frequency=10000, by_nfe=True):
        super().__init__()
        self.frequency = frequency
        self.by_nfe = by_nfe
        self.iteration = 0
        self.last_invocation = 0

    def start_run(self, algorithm):
        self.last_invocation = algorithm.nfe if self.by_nfe else self.iteration

    def post_step(self, algorithm):
        self.iteration += 1

        if self.by_nfe:
            if algorithm.nfe - self.last_invocation >= self.frequency:
                self.do_action(algorithm)
                self.last_invocation = algorithm.nfe
        else:
            if self.iteration - self.last_invocation >= self.frequency:
                self.do_action(algorithm)
                self.last_invocation = self.iteration

    @abstractmethod
    def do_action(self, algorithm):
        """Performs the action defined by this extension."""
        pass

class LoggingExtension(FixedFrequencyExtension):
    """Logs a run's progress."""

    def start_run(self, algorithm):
        super().start_run(algorithm)
        self.start_time = time.time()

        LOGGER.info("%s starting", type(algorithm).__name__)

    def end_run(self, algorithm):
        super().end_run(algorithm)

        LOGGER.info("%s finished; Total NFE: %d, Elapsed Time: %s",
                    type(algorithm).__name__,
                    algorithm.nfe,
                    datetime.timedelta(seconds=time.time()-self.start_time))

    def do_action(self, algorithm):
        LOGGER.info("%s running; NFE Complete: %d, Elapsed Time: %s",
                    type(algorithm).__name__,
                    algorithm.nfe,
                    datetime.timedelta(seconds=time.time()-self.start_time))

class AdaptiveTimeContinuationExtension(FixedFrequencyExtension):
    """Extends an algorithm to enable adaptive time continuation.

    Adaptive time continuation performs two key functions:

    First, it scales the population based on the size of the archive.  The idea
    being a larger archive, with more non-dominated solutions, requires a
    larger population to cover the search space.

    Second, it periodically introduces extra randomness or diversity into the
    population.  This helps avoid or escape local optima.

    Parameters
    ----------
    window_size : int
        The number of iterations between calls to :meth:`check`.
    max_window_size : int
        The maximum number of iterations before a restart is required.
    population_rato : float
        The ratio between the desired population size and archive size, used
        to scale the population size after each restart.
    min_population_size : int
        The minimum allowed population size.
    max_population_size : int
        The maximum allowed population size.
    mutator : Variator
        The mutation operator applied during restarts to introduce additional
        randomness or diversity into the population.  Must have an arity of
        :code:`1`.
    """

    def __init__(self,
                 window_size=100,
                 max_window_size=1000,
                 population_ratio=4.0,
                 min_population_size=10,
                 max_population_size=10000,
                 mutator=UM(1.0)):
        super().__init__(frequency=window_size,
                         by_nfe=False)
        self.window_size = window_size
        self.max_window_size = max_window_size
        self.population_ratio = population_ratio
        self.min_population_size = min_population_size
        self.max_population_size = max_population_size
        self.mutator = mutator
        self.last_restart = 0

    def check(self, algorithm):
        """Checks if a restart is required."""
        population_size = len(algorithm.population)
        target_size = self.population_ratio * len(algorithm.archive)

        if self.iteration - self.last_restart >= self.max_window_size:
            return True
        elif (target_size >= self.min_population_size and
              target_size <= self.max_population_size and
              abs(population_size - target_size) > (0.25 * target_size)):
            return True
        else:
            return False

    def restart(self, algorithm):
        """Performs the restart procedure."""
        archive = algorithm.archive
        population = archive[:]

        new_size = int(self.population_ratio * len(archive))

        if new_size < self.min_population_size:
            new_size = self.min_population_size
        elif new_size > self.max_population_size:
            new_size = self.max_population_size

        LOGGER.info("%s restarting; adjusting population size from %d to %d", type(algorithm).__name__,
                    algorithm.population_size, new_size)

        offspring = []

        while len(population) + len(offspring) < new_size:
            parents = [archive[random.randrange(len(archive))] for _ in range(self.mutator.arity)]
            offspring.extend(self.mutator.evolve(parents))

        algorithm.evaluate_all(offspring)

        population.extend(offspring)
        archive.extend(offspring)

        self.last_restart = self.iteration
        algorithm.population = population
        algorithm.population_size = len(population)

    def do_action(self, algorithm):
        if self.check(algorithm):
            self.restart(algorithm)

class EpsilonProgressContinuationExtension(AdaptiveTimeContinuationExtension):
    """Extends an algorithm to enable epsilon-progress continuation.

    Epsilon-progress continuation extends adaptive time continuation to also
    track the number of improvements made in the :class:`EpsilonBoxArchive`.
    A restart is triggered if no improvements were recorded, as that often
    occurs when the algorithm as converged to a local optima.

    Parameters
    ----------
    window_size : int
        The number of iterations between calls to :meth:`check`.
    max_window_size : int
        The maximum number of iterations before a restart is required.
    population_rato : float
        The ratio between the desired population size and archive size, used
        to scale the population size after each restart.
    min_population_size : int
        The minimum allowed population size.
    max_population_size : int
        The maximum allowed population size.
    mutator : Variator
        The mutation operator applied during restarts to introduce additional
        randomness or diversity into the population.  Must have an arity of
        :code:`1`.
    """

    def __init__(self,
                 window_size=100,
                 max_window_size=1000,
                 population_ratio=4.0,
                 min_population_size=10,
                 max_population_size=10000,
                 mutator=UM(1.0)):
        super().__init__(window_size,
                         max_window_size,
                         population_ratio,
                         min_population_size,
                         max_population_size,
                         mutator)
        self.last_improvements = 0

    def check(self, algorithm):
        result = super().check(algorithm)

        if not result:
            if algorithm.archive.improvements <= self.last_improvements:
                result = True

        self.last_improvements = algorithm.archive.improvements
        return result

    def restart(self, algorithm):
        super().restart(algorithm)
        self.last_improvements = algorithm.archive.improvements

class SaveResultsExtension(FixedFrequencyExtension):
    """Saves intermediate results to a JSON file.

    The filename pattern can reference the following variables:
    * :code:`{algorithm}` - The algorithm name
    * :code:`{problem}` - The problem name
    * :code:`{nfe}` - The number of function evaluations
    * :code:`{nvars}` - The number of variables in the problem
    * :code:`{nobjs}` - The number of objectives in the problem
    * :code:`{nconstrs}` - The number of constraints in the problem

    Parameters
    ----------
    filename_pattern: str
        The filename pattern.
    frequency : int
        The frequency the action occurs.
    by_nfe : bool
        If :code:`True`, the frequency is given in number of function
        evaluations.  If :code:`False`, the frequency is given in the number
        of iterations.
    """

    def __init__(self, filename_pattern, frequency=10000, by_nfe=True):
        super().__init__(frequency, by_nfe)
        self.filename_pattern = filename_pattern

    def do_action(self, algorithm):
        filename = self.filename_pattern.format(algorithm=type(algorithm).__name__,
                                                problem=type(algorithm.problem).__name__,
                                                nfe=algorithm.nfe,
                                                nvars=algorithm.problem.nvars,
                                                nobjs=algorithm.problem.nobjs,
                                                nconstrs=algorithm.problem.nconstrs)
        save_json(filename, algorithm.result)
