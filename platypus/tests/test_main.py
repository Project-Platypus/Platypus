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
import contextlib
import io
import os
import random
import re

from ..__main__ import main


def test_cli():
    suffix = str(random.randint(0, 1000000))

    try:
        main(["solve",
              "--algorithm", "NSGAII",
              "--problem", "DTLZ2",
              "--nfe", "10000",
              "--output", f"NSGAII_DTLZ2_{suffix}.set"])

        assert os.path.exists(f"NSGAII_DTLZ2_{suffix}.set")

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["hypervolume",
                  "--reference_set", "examples/DTLZ2.2D.pf",
                  f"NSGAII_DTLZ2_{suffix}.set"])
            assert re.match(r"[0-9]+\.[0-9]+", capture.getvalue())

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["gd",
                  "--reference_set", "examples/DTLZ2.2D.pf",
                  f"NSGAII_DTLZ2_{suffix}.set"])
            assert re.match(r"[0-9]+\.[0-9]+", capture.getvalue())

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["igd",
                  "--reference_set", "examples/DTLZ2.2D.pf",
                  f"NSGAII_DTLZ2_{suffix}.set"])
            assert re.match(r"[0-9]+\.[0-9]+", capture.getvalue())

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["epsilon",
                  "--reference_set", "examples/DTLZ2.2D.pf",
                  f"NSGAII_DTLZ2_{suffix}.set"])
            assert re.match(r"[0-9]+\.[0-9]+", capture.getvalue())

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["spacing",
                  f"NSGAII_DTLZ2_{suffix}.set"])
            re.match(r"[0-9]+\.[0-9]+", capture.getvalue())

        main(["filter",
              "--unique",
              "--feasible",
              "--nondominated",
              "--output", f"NSGAII_DTLZ2_filtered_{suffix}.set",
              f"NSGAII_DTLZ2_{suffix}.set"])

        assert os.path.exists(f"NSGAII_DTLZ2_filtered_{suffix}.set")

        main(["normalize",
              "--reference_set", "examples/DTLZ2.2D.pf",
              "--output", f"NSGAII_DTLZ2_normalized_{suffix}.set",
              f"NSGAII_DTLZ2_{suffix}.set"])

        assert os.path.exists(f"NSGAII_DTLZ2_normalized_{suffix}.set")

        main(["plot",
              "--output", f"NSGAII_DTLZ2_{suffix}.png",
              f"NSGAII_DTLZ2_{suffix}.set"])

        assert os.path.exists(f"NSGAII_DTLZ2_{suffix}.png")
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(f"NSGAII_DTLZ2_{suffix}.set")
            os.remove(f"NSGAII_DTLZ2_filtered_{suffix}.set")
            os.remove(f"NSGAII_DTLZ2_normalized_{suffix}.set")
            os.remove(f"NSGAII_DTLZ2_{suffix}.png")
