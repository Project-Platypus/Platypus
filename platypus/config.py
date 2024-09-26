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

import inspect

from ._tools import coalesce
from .errors import PlatypusError
from .types import Type


class _PlatypusConfig:
    """Global configuration options for Platypus."""

    def __init__(self):
        super().__init__()
        self._version = None
        self._default_variator = {}
        self._default_mutator = {}
        self._default_logger = None
        self.default_evaluator = None
        self.default_log_frequency = None

    @property
    def version(self):
        return self._version

    def register_default_variator(self, type, operator):
        """Registers or overwrites the default variator.

        Parameters
        ----------
        type : Type
            The type of decision variable.
        operator : Variator
            The default variator to associated with the type.
        """
        self._default_variator[type] = operator

    def register_default_mutator(self, type, operator):
        """Registers or overwrites the default mutator.

        Parameters
        ----------
        type : Type
            The type of decision variable.
        operator : Mutation
            The default mutator to associated with the type.
        """
        self._default_mutator[type] = operator

    def default_variator(self, problem):
        """Returns the default variator for the given type or problem.

        Variators must first be registered by :func:`register_default_variator`
        to be discovered by this method.

        Parameters
        ----------
        problem : Type or Problem
            The type or problem used to select the default variator.
        """
        if inspect.isclass(problem) and issubclass(problem, Type):
            base_type = problem
        else:
            if len(problem.types) == 0:
                raise PlatypusError("problem has no decision variables")

            base_type = problem.types[0].__class__

            if not all([isinstance(t, base_type) for t in problem.types]):
                raise PlatypusError("must explicitly set variator for mixed types")

        if base_type in self._default_variator:
            return self._default_variator[base_type]

        for default_type in self._default_variator.keys():
            if issubclass(base_type, default_type):
                return self._default_variator[default_type]

        raise PlatypusError(f"no default variator for {base_type}")

    def default_mutator(self, problem):
        """Returns the default mutator for the given type or problem.

        Mutators must first be registered by :func:`register_default_mutator`
        to be discovered by this method.

        Parameters
        ----------
        problem : Type or Problem
            The type or problem used to select the default mutator.
        """
        if inspect.isclass(problem) and issubclass(problem, Type):
            base_type = problem
        else:
            if len(problem.types) == 0:
                raise PlatypusError("problem has no decision variables")

            base_type = problem.types[0].__class__

            if not all([isinstance(t, base_type) for t in problem.types]):
                raise PlatypusError("must explicitly set mutator for mixed types")

        if base_type in self._default_mutator:
            return self._default_mutator[base_type]

        for default_type in self._default_mutator.keys():
            if issubclass(base_type, default_type):
                return self._default_mutator[default_type]

        raise PlatypusError(f"no default mutator for {base_type}")

    def get_logging_extension(self, log_frequency=None):
        """Returns the default logging extension.

        Parameters
        ----------
        log_frequency : int
            The frequency that run progress log messages are written.  If
            :code:`None`, uses :code:`default_log_frequency`.
        """
        return self._default_logger(coalesce(log_frequency, self.default_log_frequency, 0))


# Defaults are configured in __init__.py.
PlatypusConfig = _PlatypusConfig()
