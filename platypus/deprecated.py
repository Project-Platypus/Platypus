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

import warnings
from .core import nondominated_sort_cmp
from .config import PlatypusConfig

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
