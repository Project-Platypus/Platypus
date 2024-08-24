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

import sys
import time
import datetime
import functools
from collections import OrderedDict
from .config import PlatypusConfig
from .errors import PlatypusError
from .evaluator import Job

class ExperimentJob(Job):

    def __init__(self, instance, nfe, algorithm_name, problem_name, seed, display_stats):
        super().__init__()
        self.instance = instance
        self.nfe = nfe
        self.algorithm_name = algorithm_name
        self.problem_name = problem_name
        self.seed = seed
        self.display_stats = display_stats

    def run(self):
        if self.display_stats:
            start_time = time.time()
            print(f"Running seed {self.seed} of {self.algorithm_name} on {self.problem_name}")

        self.instance.run(self.nfe)

        if self.display_stats:
            end_time = time.time()
            print(f"Finished seed {self.seed} of {self.algorithm_name} on {self.problem_name} in "
                  f"{datetime.timedelta(seconds=round(end_time-start_time))}")

class IndicatorJob(Job):

    def __init__(self, algorithm_name, problem_name, result_set, indicators):
        super().__init__()
        self.algorithm_name = algorithm_name
        self.problem_name = problem_name
        self.result_set = result_set
        self.indicators = indicators

    def run(self):
        self.results = [indicator(self.result_set) for indicator in self.indicators]

def evaluate_job_generator(algorithms, problems, seeds, nfe, display_stats):
    existing_algorithms = set()
    existing_problems = set()

    for i in range(len(algorithms)):
        if isinstance(algorithms[i], tuple):
            algorithm = algorithms[i][0]

            if len(algorithms[i]) >= 2:
                kwargs = algorithms[i][1]
            else:
                kwargs = {}

            if len(algorithms[i]) >= 3:
                algorithm_name = algorithms[i][2]
            else:
                algorithm_name = algorithm.__name__

        else:
            algorithm = algorithms[i]
            algorithm_name = algorithm.__name__
            kwargs = {}

        if algorithm_name in existing_algorithms:
            raise PlatypusError("only one algorithm with name " + algorithm_name + " can be run")
        else:
            existing_algorithms.add(algorithm_name)

        for j in range(len(problems)):
            if isinstance(problems[j], tuple):
                problem = problems[j][0]

                if isinstance(problem, type):
                    problem = problem()

                if len(problems[j]) >= 2:
                    problem_name = problems[j][1]
                else:
                    problem_name = problem.__class__.__name__
            else:
                problem = problems[j]

                if isinstance(problem, type):
                    problem = problem()

                problem_name = problem.__class__.__name__

            if i == 0:
                if problem_name in existing_problems:
                    raise PlatypusError("only one problem with name " + problem_name + " can be run")
                else:
                    existing_problems.add(problem_name)

            for k in range(seeds):
                yield ExperimentJob(algorithm(problem, **kwargs),
                                    nfe,
                                    algorithm_name,
                                    problem_name,
                                    k,
                                    display_stats)

def experiment(algorithms=[],
               problems=[],
               seeds=10,
               nfe=10000,
               evaluator=None,
               display_stats=False):
    """Run experiments.

    Used to run experiments where one or more algorithms are tested on one or
    more problems.  Returns a dict containing the results.  The dict is of
    the form:
        pareto_set = result["algorithm"]["problem"][seed_index]

    Parameters
    ----------
    algorithms : list
        List of algorithms to run.  Can either be a type of Algorithm or a
        tuple defining ``(type, kwargs, name)``, where type is the Algorithm's
        type, kwargs is a dict defining any optional parameters for the
        algorithm, and name is a human-readable name for the algorithm.  All
        algorithms must have unique names.  If a name is not provided, the
        type name is used.
    problems : list
        List of problems to run.  Can either be a type of Problem, an instance
        of a Problem, or a tuple defining ``(type, name)``, where type is the
        Problem's type and name is a human-readable name for the problem.  All
        problems must have unique names.  If a name is not provided, the type
        name is used.
    seeds : int
        The number of replicates of each experiment to run
    nfe : int
        The number of function evaluations allotted to each experiment
    display_stats : bool
        If True, the progress of the experiments is output to the screen
    """
    if not isinstance(algorithms, list):
        algorithms = [algorithms]

    if not isinstance(problems, list):
        problems = [problems]

    # construct the jobs to run
    generator = evaluate_job_generator(algorithms, problems, seeds, nfe, display_stats)

    # process the jobs
    if evaluator is None:
        evaluator = PlatypusConfig.default_evaluator

    job_results = evaluator.evaluate_all(generator)

    # convert results to structured format
    results = OrderedDict()
    count = 0

    for job in job_results:
        if job.algorithm_name not in results:
            results[job.algorithm_name] = {}

        if job.problem_name not in results[job.algorithm_name]:
            results[job.algorithm_name][job.problem_name] = []

        results[job.algorithm_name][job.problem_name].append(job.instance.result)
        count += 1

    return results

def calculate_job_generator(results, indicators):
    for algorithm in results.keys():
        for problem in results[algorithm].keys():
            for result_set in results[algorithm][problem]:
                yield IndicatorJob(algorithm, problem, result_set, indicators)

def calculate(results,
              indicators=[],
              evaluator=None):
    if not isinstance(indicators, list):
        indicators = [indicators]

    if evaluator is None:
        evaluator = PlatypusConfig.default_evaluator

    generator = calculate_job_generator(results, indicators)
    indicator_results = evaluator.evaluate_all(generator)

    results = OrderedDict()

    for job in indicator_results:
        if job.algorithm_name not in results:
            results[job.algorithm_name] = {}

        if job.problem_name not in results[job.algorithm_name]:
            results[job.algorithm_name][job.problem_name] = {}

        for i in range(len(indicators)):
            indicator_name = indicators[i].__class__.__name__

            if indicator_name not in results[job.algorithm_name][job.problem_name]:
                results[job.algorithm_name][job.problem_name][indicator_name] = []

            results[job.algorithm_name][job.problem_name][indicator_name].append(job.results[i])

    return results

def display(results, ndigits=None, file=sys.stdout):
    for algorithm in results.keys():
        print(f"{algorithm}:", file=file)
        for problem in results[algorithm].keys():
            if isinstance(results[algorithm][problem], dict):
                print(f"    {problem}:")
                for indicator in results[algorithm][problem].keys():
                    if ndigits:
                        print(f"        {indicator}: {list(map(functools.partial(round, ndigits=ndigits), results[algorithm][problem][indicator]))}", file=file)
                    else:
                        print(f"        {indicator}: {results[algorithm][problem][indicator]}", file=file)
            else:
                print(f"    {problem}: {results[algorithm][problem]}", file=file)
