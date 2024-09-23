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

class PlatypusWarning(UserWarning):
    """Warning about a potential issue with the usage of Platypus.

    Warnings emitted by Platypus should extend from this type, unless a more
    standard warning type is available (e.g., :class:`DeprecationWarning`).
    These warnings typically indicate incorrect usage that could potentially
    lead to issues, but not at the severity of raising an exception.
    """
    pass

class PlatypusError(Exception):
    """An exception occurred in the Platypus library.

    Exceptions raised by Platypus should extend from this type, unless a more
    standard exception type is available (e.g., :class:`ValueError` to
    indicate a parameter is invalid).
    """
    pass

class SingularError(PlatypusError):
    """The matrix is singular and the operation could not be performed."""
    pass
