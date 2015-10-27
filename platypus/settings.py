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
from .types import Real, Binary, Permutation
from .operators import GAOperator, CompoundOperator, CompoundMutation, SBX, PM, HUX, BitFlip, PMX, Insertion, Swap
from .core import PlatypusError

class PlatypusSettings(object):
    
    default_variator = {Real : GAOperator(SBX(), PM()),
                        Binary : GAOperator(HUX(), BitFlip()),
                        Permutation : CompoundOperator(PMX(), Insertion(), Swap())}
    
    default_mutator = {Real : PM(),
                       Binary : BitFlip(),
                       Permutation : CompoundMutation(Insertion(), Swap())}
    
def default_variator(problem):
    base_type = problem.types[0].__class__
    
    if all([isinstance(t, base_type)] for t in problem.types):
        if base_type in PlatypusSettings.default_variator:
            return PlatypusSettings.default_variator[base_type]
        else:
            raise PlatypusError("no default variator for %s" % base_type)
    else:
        raise PlatypusError("must define variator for mixed types")
    
def default_mutator(problem):
    base_type = problem.types[0].__class__
    
    if all([isinstance(t, base_type)] for t in problem.types):
        if base_type in PlatypusSettings.default_mutator:
            return PlatypusSettings.default_mutator[base_type]
        else:
            raise PlatypusError("no default mutator for %s" % base_type)
    else:
        raise PlatypusError("must define mutator for mixed types") 