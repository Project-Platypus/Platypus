# Copyright 2015-2023 David Hadka
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

from .types import Real, Binary, Permutation, Subset
from .operators import GAOperator, CompoundOperator, CompoundMutation, SBX, PM, HUX, BitFlip, PMX, Insertion, Swap, SSX, Replace
from .core import PlatypusError
from .evaluator import MapEvaluator

class _PlatypusConfig:

    def __init__(self):
        super().__init__()

        self.default_variator = {Real : GAOperator(SBX(), PM()),
                                 Binary : GAOperator(HUX(), BitFlip()),
                                 Permutation : CompoundOperator(PMX(), Insertion(), Swap()),
                                 Subset : GAOperator(SSX(), Replace())}

        self.default_mutator = {Real : PM(),
                                Binary : BitFlip(),
                                Permutation : CompoundMutation(Insertion(), Swap()),
                                Subset : Replace()}

        self.default_evaluator = MapEvaluator()

        self.default_log_frequency = None

PlatypusConfig = _PlatypusConfig()

def default_variator(problem):
    if len(problem.types) == 0:
        raise PlatypusError("problem has no decision variables")

    base_type = problem.types[0].__class__

    if all([isinstance(t, base_type) for t in problem.types]):
        if base_type in PlatypusConfig.default_variator:
            return PlatypusConfig.default_variator[base_type]
        else:
            for default_type in PlatypusConfig.default_variator.keys():
                if issubclass(base_type, default_type):
                    return PlatypusConfig.default_variator[default_type]

            raise PlatypusError(f"no default variator for {base_type}")
    else:
        raise PlatypusError("must define variator for mixed types")

def default_mutator(problem):
    if len(problem.types) == 0:
        raise PlatypusError("problem has no decision variables")

    base_type = problem.types[0].__class__

    if all([isinstance(t, base_type) for t in problem.types]):
        if base_type in PlatypusConfig.default_mutator:
            return PlatypusConfig.default_mutator[base_type]
        else:
            for default_type in PlatypusConfig.default_mutator.keys():
                if issubclass(base_type, default_type):
                    return PlatypusConfig.default_mutator[default_type]

            raise PlatypusError(f"no default mutator for {base_type}")
    else:
        raise PlatypusError("must define mutator for mixed types")
