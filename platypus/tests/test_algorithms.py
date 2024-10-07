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
import jsonpickle
import pytest

from ..algorithms import (CMAES, GDE3, IBEA, MOEAD, NSGAII, NSGAIII, OMOPSO,
                          SMPSO, SPEA2, AbstractGeneticAlgorithm, EpsMOEA,
                          EpsNSGAII, ParticleSwarm)
from ..core import Direction, Problem
from ..errors import PlatypusError
from ..extensions import LoggingExtension
from ..problems import CF1, DTLZ2
from ..weights import normal_boundary_weights, pbi
from ._utils import similar

problems = [DTLZ2, CF1]

algorithms = {
    NSGAII: [{}],
    NSGAIII: [{"divisions_outer": 24},
              {"divisions_outer": 1, "divisions_inner": 4}],
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
    EpsNSGAII: [{"epsilons": [0.01]}],
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
    similar(algorithm, copy)

@pytest.mark.parametrize("algorithm", create_instances())
def test_jsonpickle(algorithm):
    s = jsonpickle.dumps(algorithm)
    copy = jsonpickle.loads(s)
    similar(algorithm, copy)

@pytest.mark.parametrize("algorithm", create_instances())
def test_run(algorithm):
    assert algorithm.nfe == 0
    algorithm.run(500)
    assert algorithm.nfe >= 500

    if isinstance(algorithm, AbstractGeneticAlgorithm):
        assert algorithm.nfe % algorithm.population_size == 0
    elif isinstance(algorithm, ParticleSwarm):
        assert algorithm.nfe % algorithm.swarm_size == 0
    elif isinstance(algorithm, CMAES):
        assert algorithm.nfe % algorithm.offspring_size == 0

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

def test_extensions():
    algorithm = NSGAII(DTLZ2())

    # always start with the LoggingExtension
    assert len(algorithm._extensions) == 1

    algorithm.remove_extension(LoggingExtension)
    assert len(algorithm._extensions) == 0

    l1 = LoggingExtension()
    l2 = LoggingExtension()

    algorithm.add_extension(l1)
    algorithm.add_extension(l2)
    assert [l2, l1] == algorithm._extensions

    algorithm.remove_extension(l2)
    assert [l1] == algorithm._extensions
