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

from .algorithms import (CMAES, GDE3, IBEA, MOEAD, NSGAII, NSGAIII, OMOPSO,
                         PAES, PESA2, SMPSO, SPEA2, AbstractGeneticAlgorithm,
                         EpsMOEA, EpsNSGAII, EvolutionaryStrategy,
                         GeneticAlgorithm, ParticleSwarm, RegionBasedSelector,
                         SingleObjectiveAlgorithm)
from .config import PlatypusConfig
from .core import (AdaptiveGridArchive, Algorithm, Archive, AttributeDominance,
                   Constraint, Direction, Dominance, EpsilonBoxArchive,
                   EpsilonDominance, FitnessArchive, FitnessEvaluator,
                   FixedLengthArray, Generator, HypervolumeFitnessEvaluator,
                   Indicator, MaxEvaluations, MaxTime, Mutation,
                   ParetoDominance, Problem, Selector, Solution,
                   TerminationCondition, Variator, crowding_distance,
                   nondominated, nondominated_prune, nondominated_sort,
                   nondominated_sort_cmp, nondominated_split,
                   nondominated_truncate, normalize, truncate_fitness)
from .deprecated import (AdaptiveTimeContinuation, EpsilonProgressContinuation,
                         PeriodicAction, default_mutator, default_variator,
                         nondominated_cmp)
from .errors import PlatypusError
from .evaluator import (ApplyEvaluator, Evaluator, Job, MapEvaluator,
                        MultiprocessingEvaluator, PoolEvaluator,
                        ProcessPoolEvaluator, SubmitEvaluator)
from .experimenter import (ExperimentJob, IndicatorJob, calculate, display,
                           experiment)
from .extensions import (AdaptiveTimeContinuationExtension,
                         EpsilonProgressContinuationExtension, Extension,
                         FixedFrequencyExtension, LoggingExtension,
                         SaveResultsExtension)
from .filters import (crowding_distance_key, feasible, fitness_key, infeasible,
                      matches, objectives_key, rank_key, truncate, unique,
                      variables_key)
from .indicators import (EpsilonIndicator, GenerationalDistance, Hypervolume,
                         InvertedGenerationalDistance, Spacing)
from .io import (dump, load, load_json, load_objectives, load_state, save_json,
                 save_objectives, save_state)
from .operators import (HUX, PCX, PM, PMX, SBX, SPX, SSX, UM, UNDX, BitFlip,
                        CompoundMutation, CompoundOperator,
                        DifferentialEvolution, GAOperator, InjectedPopulation,
                        Insertion, Multimethod, NonUniformMutation,
                        RandomGenerator, Replace, Swap, TournamentSelector,
                        UniformMutation)
from .problems import (CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10,
                       DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7, UF1, UF2, UF3, UF4,
                       UF5, UF6, UF7, UF8, UF9, UF10, UF11, UF12, UF13, WFG,
                       WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9,
                       ZDT, ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6)
from .types import Binary, Integer, Permutation, Real, Subset, Type
from .weights import chebyshev, normal_boundary_weights, pbi, random_weights

__version__ = "1.4.1"

PlatypusConfig.register_default_variator(Real, GAOperator(SBX(), PM()))
PlatypusConfig.register_default_variator(Binary, GAOperator(HUX(), BitFlip()))
PlatypusConfig.register_default_variator(Permutation, CompoundOperator(PMX(), Insertion(), Swap()))
PlatypusConfig.register_default_variator(Subset, GAOperator(SSX(), Replace()))

PlatypusConfig.register_default_mutator(Real, PM())
PlatypusConfig.register_default_mutator(Binary, BitFlip())
PlatypusConfig.register_default_mutator(Permutation, CompoundMutation(Insertion(), Swap()))
PlatypusConfig.register_default_mutator(Subset, Replace())

PlatypusConfig._default_logger = LoggingExtension
PlatypusConfig.default_evaluator = MapEvaluator()
PlatypusConfig._version = __version__
