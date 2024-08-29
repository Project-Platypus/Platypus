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

from .core import FixedLengthArray, Problem, Generator, Variator, \
    Mutation, Selector, TerminationCondition, MaxEvaluations, MaxTime, \
    Algorithm, Constraint, Solution, Dominance, ParetoDominance, \
    EpsilonDominance, AttributeDominance, Archive, AdaptiveGridArchive, \
    FitnessArchive, EpsilonBoxArchive, nondominated, crowding_distance, \
    nondominated_sort_cmp, nondominated_sort, nondominated_split, \
    nondominated_prune, nondominated_truncate, truncate_fitness, \
    normalize, FitnessEvaluator, HypervolumeFitnessEvaluator, Indicator

from .errors import PlatypusError

from .config import PlatypusConfig

from .filters import unique, truncate, matches, feasible, infeasible, \
    objectives_key, variables_key, fitness_key, rank_key, \
    crowding_distance_key

from .algorithms import AbstractGeneticAlgorithm, SingleObjectiveAlgorithm, \
    GeneticAlgorithm, EvolutionaryStrategy, NSGAII, EpsMOEA, GDE3, SPEA2, \
    MOEAD, NSGAIII, ParticleSwarm, OMOPSO, SMPSO, CMAES, IBEA, PAES, \
    RegionBasedSelector, PESA2, PeriodicAction, AdaptiveTimeContinuation, \
    EpsilonProgressContinuation, EpsNSGAII

from .evaluator import Job, Evaluator, MapEvaluator, SubmitEvaluator, \
    ApplyEvaluator, PoolEvaluator, MultiprocessingEvaluator, \
    ProcessPoolEvaluator

from .experimenter import ExperimentJob, IndicatorJob, experiment, calculate, \
    display

from .indicators import GenerationalDistance, InvertedGenerationalDistance, \
    EpsilonIndicator, Spacing, Hypervolume

from .operators import RandomGenerator, InjectedPopulation, \
    TournamentSelector, PM, SBX, GAOperator, CompoundMutation, \
    CompoundOperator, DifferentialEvolution, UniformMutation, \
    NonUniformMutation, UM, PCX, UNDX, SPX, BitFlip, HUX, Swap, PMX, \
    Insertion, Replace, SSX, Multimethod

from .problems import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7, WFG, WFG1, WFG2, \
    WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9, UF1, UF2, UF3, UF4, UF5, UF6, \
    UF7, UF8, UF9, UF10, UF11, UF12, UF13, CF1, CF2, CF3, CF4, CF5, CF6, CF7, \
    CF8, CF9, CF10, ZDT, ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6

from .types import Type, Real, Binary, Integer, Permutation, Subset

from .weights import chebyshev, pbi, random_weights, normal_boundary_weights

from .io import save_objectives, load_objectives, save_json

PlatypusConfig.register_default_variator(Real, GAOperator(SBX(), PM()))
PlatypusConfig.register_default_variator(Binary, GAOperator(HUX(), BitFlip()))
PlatypusConfig.register_default_variator(Permutation, CompoundOperator(PMX(), Insertion(), Swap()))
PlatypusConfig.register_default_variator(Subset, GAOperator(SSX(), Replace()))

PlatypusConfig.register_default_mutator(Real, PM())
PlatypusConfig.register_default_mutator(Binary, BitFlip())
PlatypusConfig.register_default_mutator(Permutation, CompoundMutation(Insertion(), Swap()))
PlatypusConfig.register_default_mutator(Subset, Replace())

PlatypusConfig.default_evaluator = MapEvaluator()

__version__ = "1.3.0"
