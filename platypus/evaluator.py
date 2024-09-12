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

import time
import logging
import datetime
from abc import ABCMeta, abstractmethod

LOGGER = logging.getLogger("Platypus")

def _chunks(items, n):
    """Splits a list into fixed-sized chunks.

    Parameters
    ----------
    items : iterable
        The list of items to split.
    n : int
        The size of each chunk.
    """
    result = []
    iterator = iter(items)

    try:
        while True:
            result.append(next(iterator))

            if len(result) == n:
                yield result
                result = []
    except StopIteration:
        if len(result) > 0:
            yield result

class Job(metaclass=ABCMeta):
    """Abstract class for implementing a distributable job.

    The job should capture any inputs required by :meth:`run` along with any
    outputs produced by the job.

    Also be aware that the specific :class:`Evaluator` used to run the jobs
    might mandate additional requirements.  For instance, evaluators that
    distributed jobs across processes or machines typically need to
    serialize or pickle Python objects to transmit them over a network.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(self):
        """Executes the job."""
        pass

def run_job(job):
    job.run()
    return job

class Evaluator(metaclass=ABCMeta):
    """Abstract class for evaluators."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate_all(self, jobs, **kwargs):
        """Evaluates all of the jobs.

        Parameters
        ----------
        jobs : iterable of Job
            The jobs to execute.
        kwargs :
            Any additional arguments passed on to the evaluator.

        Returns
        -------
        The evaluated jobs.
        """
        raise NotImplementedError()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class MapEvaluator(Evaluator):
    """Evaluates jobs using a given map-like function.

    When used with the built-in :meth:`map` function, jobs are essentially
    run sequentially on the local machine.

    Parameters
    ----------
    map_func : Callable
        The map-like function.
    """

    def __init__(self, map_func=map):
        super().__init__()
        self.map_func = map_func

    def evaluate_all(self, jobs, **kwargs):
        log_frequency = kwargs.get("log_frequency", None)

        if log_frequency is None:
            return list(self.map_func(run_job, jobs))
        else:
            result = []
            job_name = kwargs.get("job_name", "Batch Jobs")
            start_time = time.time()

            for chunk in _chunks(jobs, log_frequency):
                result.extend(self.map_func(run_job, chunk))
                LOGGER.log(logging.INFO,
                           "%s running; Jobs Complete: %d, Elapsed Time: %s",
                           job_name,
                           len(result),
                           datetime.timedelta(seconds=time.time()-start_time))

            return result

class SubmitEvaluator(Evaluator):
    """Evaluates jobs using a given submit function.

    A submit function is a form of asynchronous computing that takes a
    callable and a list of inputs, asynchronously evaluates the function on
    each input, and returns a list of futures to await the results.

    Thus, the submit function should satisfy this signature::

        def submit(func: Callable[[T], R], inputs: list[T]) -> list[Future[R]]:
            ...

    where :code:`func` takes input of type :code:`T` and returns a result of
    type :code:`R`.  In the context of an :class:`Evaluator`, both types will
    be :class:`Job`.

    For more information, see the :mod:`concurrent.futures` module and, in
    particular, the :class:`Executor` class.

    Parameters
    ----------
    submit_func : Callable
        The submit function.
    """

    def __init__(self, submit_func):
        super().__init__()
        self.submit_func = submit_func

    def evaluate_all(self, jobs, **kwargs):
        futures = [self.submit_func(run_job, job) for job in jobs]
        log_frequency = kwargs.get("log_frequency", None)

        if log_frequency is None:
            return [f.result() for f in futures]
        else:
            result = []
            job_name = kwargs.get("job_name", "Batch Jobs")
            start_time = time.time()

            for chunk in _chunks(futures, log_frequency):
                result.extend([f.result() for f in chunk])
                LOGGER.log(logging.INFO,
                           "%s running; Jobs Complete: %d, Elapsed Time: %s",
                           job_name,
                           len(result),
                           datetime.timedelta(seconds=time.time()-start_time))

            return result

class ApplyEvaluator(Evaluator):
    """Evaluates jobs using a given apply function.

    An apply function is a form of asynchronous computing that takes a
    callable and a input, asynchronously evaluates the function on that input,
    and returns a future to await the result.

    Thus, the apply function should satisfy this signature::

        def apply(func: Callable[[T], R], input: T) -> Future[R]:
            ...

    where :code:`func` takes input of type :code:`T` and returns a result of
    type :code:`R`.  In the context of an :class:`Evaluator`, both types will
    be :class:`Job`.

    Thus, the main difference between a submit and an apply function is
    whether it accepts a single input or a list of inputs.

    For an example, see the :mod:`multiprocessing` module and, in particular,
    the :class:`Pool` class, which provides both :code:`apply` and :code:`map`
    functions.

    Parameters
    ----------
    apply_func : Callable
        The apply function.
    """

    def __init__(self, apply_func):
        super().__init__()
        self.apply_func = apply_func

    def evaluate_all(self, jobs, **kwargs):
        futures = [self.apply_func(run_job, [job]) for job in jobs]
        log_frequency = kwargs.get("log_frequency", None)

        if log_frequency is None:
            return [f.get() for f in futures]
        else:
            result = []
            job_name = kwargs.get("job_name", "Batch Jobs")
            start_time = time.time()

            for chunk in _chunks(futures, log_frequency):
                result.extend([f.get() for f in chunk])
                LOGGER.log(logging.INFO,
                           "%s running; Jobs Complete: %d, Elapsed Time: %s",
                           job_name,
                           len(result),
                           datetime.timedelta(seconds=time.time()-start_time))

            return result

class PoolEvaluator(MapEvaluator):
    """Evaluates jobs using a pool.

    The two most common pool implementations are :code:`MPIPool`, which is
    included with Platypus, and :code:`Schwimmbad`.  In order to be considered
    a pool, the implementation must:

    1. Provide the :code:`map` attribute with a map-like function for submitting
       jobs to the pool.
    2. Provide a :code:`close()` method that stops accepting new jobs and
       begins shutting down the pool.
    3. As jobs can continue running after closing a pool, optionally provide
       a :code:`join()` method to wait for the completion of all jobs.

    Parameters
    ----------
    pool : Any
        The pool.
    """

    def __init__(self, pool):
        super().__init__(pool.map)
        self.pool = pool

        if hasattr(pool, "_processes"):
            LOGGER.log(logging.INFO, "Started pool evaluator with %d processes", pool._processes)
        else:
            LOGGER.log(logging.INFO, "Started pool evaluator")

    def close(self):
        LOGGER.log(logging.DEBUG, "Closing pool evaluator")
        self.pool.close()

        if hasattr(self.pool, "join"):
            LOGGER.log(logging.DEBUG, "Waiting for all processes to complete")
            self.pool.join()

        LOGGER.log(logging.INFO, "Closed pool evaluator")

class MultiprocessingEvaluator(PoolEvaluator):
    """Evaluator using Python's multiprocessing library.

    Parallelization is provided by spawning multiple Python processes.  Refer
    to the :mod:`multiprocessing` module and :class:`Pool` for any additional
    requirements.

    Parameters
    ----------
    processes : int
        The number of processes to spawn.  If :code:`None`, will use the number
        of available CPUs.
    """

    def __init__(self, processes=None):
        try:
            from multiprocessing import Pool
            super().__init__(Pool(processes))
        except ImportError:
            # prevent error from showing in Eclipse if multiprocessing not available
            raise

class ProcessPoolEvaluator(SubmitEvaluator):
    """Evaluator using Python's ProcessPoolExecutor.

    Refer to the :mod:`concurrent.futures` module and
    :class:`ProcessPoolExecutor`for any additional requirements.

    Parameters
    ----------
    processes : int
        The size of the process pool.  If :code:`None`, will use the number
        of available CPUs.
    """

    def __init__(self, processes=None):
        try:
            from concurrent.futures import ProcessPoolExecutor
            self.executor = ProcessPoolExecutor(processes)
            super().__init__(self.executor.submit)
            LOGGER.log(logging.INFO, "Started process pool evaluator")

            if processes:
                LOGGER.log(logging.INFO, "Using user-defined number of processes: %d", processes)
        except ImportError:
            # prevent error from showing in Eclipse if concurrent.futures not available
            raise

    def close(self):
        LOGGER.log(logging.DEBUG, "Closing process pool evaluator")
        self.executor.shutdown()
        LOGGER.log(logging.INFO, "Closed process pool evaluator")
