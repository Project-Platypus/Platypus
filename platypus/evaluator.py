# Copyright 2015-2018 David Hadka
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
from __future__ import absolute_import, division, print_function

import time
import logging
import datetime
from abc import ABCMeta, abstractmethod

LOGGER = logging.getLogger("Platypus")

def _chunks(items, n):
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

class Job(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Job, self).__init__()
        
    @abstractmethod
    def run(self):
        raise NotImplementedError("method not implemented")
    
def run_job(job):
    job.run()
    return job

class Evaluator(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Evaluator, self).__init__()
    
    @abstractmethod
    def evaluate_all(self, jobs, **kwargs):
        raise NotImplementedError("method not implemented")
    
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
class MapEvaluator(Evaluator):
    
    def __init__(self, map_func=map):
        super(MapEvaluator, self).__init__()
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
    
    def __init__(self, submit_func):
        super(SubmitEvaluator, self).__init__()
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
    
    def __init__(self, apply_func):
        super(ApplyEvaluator, self).__init__()
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
   
# Note: this is compatible with MPIPool and Schwimmbad 
class PoolEvaluator(MapEvaluator):
    
    def __init__(self, pool):
        super(PoolEvaluator, self).__init__(pool.map)
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
    
    def __init__(self, processes=None):
        try:
            from multiprocessing import Pool
            super(MultiprocessingEvaluator, self).__init__(Pool(processes))
        except ImportError:
            # prevent error from showing in Eclipse if multiprocessing not available
            raise

class ProcessPoolEvaluator(SubmitEvaluator):
    
    def __init__(self, processes=None):
        try:
            from concurrent.futures import ProcessPoolExecutor
            self.executor = ProcessPoolExecutor(processes)
            super(ProcessPoolEvaluator, self).__init__(self.executor.submit)
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
   
