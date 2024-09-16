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
import os
import random
import unittest
from ..__main__ import main

class TestMain(unittest.TestCase):

    def test_cli(self):
        suffix = str(random.randint(0, 1000000))

        main("solve",
             "--algorithm", "NSGAII",
             "--problem", "DTLZ2",
             "--nfe", "10000",
             "--output", f"NSGAII_DTLZ2_{suffix}.set")

        self.assertTrue(os.path.exists(f"NSGAII_DTLZ2_{suffix}.set"))

        main("hypervolume",
             "--reference", "examples/DTLZ2.2D.pf",
             f"NSGAII_DTLZ2_{suffix}.set")

        main("plot",
             "--output", f"NSGAII_DTLZ2_{suffix}.png",
             f"NSGAII_DTLZ2_{suffix}.set")

        self.assertTrue(os.path.exists(f"NSGAII_DTLZ2_{suffix}.png"))

        os.remove(f"NSGAII_DTLZ2_{suffix}.set")
        os.remove(f"NSGAII_DTLZ2_{suffix}.png")
