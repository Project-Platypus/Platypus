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
import pickle
import pytest
import jsonpickle
from ..core import Problem, Direction
from ..errors import PlatypusError
from ..problems import DTLZ2, CF1
from ..algorithms import NSGAII, NSGAIII, CMAES, GDE3, IBEA, MOEAD, OMOPSO, \
    SMPSO, SPEA2, EpsMOEA
from ..weights import normal_boundary_weights, pbi

problems = [DTLZ2, CF1]

algorithms = {
    NSGAII: [{}],
    NSGAIII: [{"divisions_outer": 24},
              {"divisions_outer": 4, "divisions_inner": 2}],
    CMAES: [{}],
    GDE3: [{}],
    IBEA: [{}],
    MOEAD: [{},
            {"weight_generator": normal_boundary_weights, "divisions_outer": 24},
            {"scalarizing_function": pbi}],
    OMOPSO: [{"epsilons": [0.01]}],
    SMPSO: [{}],
    SPEA2: [{}],
    EpsMOEA: [{"epsilons": [0.01]}],
}

constraints_not_supported = {IBEA}

def create_instances():
    for problem in problems:
        for algorithm in algorithms.keys():
            for i, kwargs in enumerate(algorithms[algorithm]):
                id = f"{problem.__name__}-{algorithm.__name__}-{i+1}"
                p = problem()

                if algorithm in constraints_not_supported and p.nconstrs > 0:
                    continue

                yield pytest.param(algorithm(p, **kwargs), id=id)

@pytest.mark.parametrize("algorithm", create_instances())
def test_pickle(algorithm):
    s = pickle.dumps(algorithm)
    copy = pickle.loads(s)
    assert type(algorithm) is type(copy)

@pytest.mark.parametrize("algorithm", create_instances())
def test_jsonpickle(algorithm):
    s = jsonpickle.dumps(algorithm)
    copy = jsonpickle.loads(s)
    assert type(algorithm) is type(copy)

@pytest.mark.parametrize("algorithm", create_instances())
def test_run(algorithm):
    algorithm.run(500)
    assert algorithm.nfe >= 500 and algorithm.nfe < 600
    assert len(algorithm.result) > 0

@pytest.fixture
def maximized_problem():
    problem = Problem(1, 1)
    problem.directions[:] = Direction.MAXIMIZE
    return problem

@pytest.mark.parametrize("algorithm", [MOEAD, NSGAIII])
def test_fail_maximization(maximized_problem, algorithm):
    with pytest.raises(PlatypusError):
        algorithm(maximized_problem, **algorithms[algorithm][0])
