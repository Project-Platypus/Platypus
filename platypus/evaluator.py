# Copyright 2015-2016 David Hadka
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

from abc import ABCMeta, abstractmethod
from multiprocessing import Pool

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
    def evaluate_all(self, jobs):
        raise NotImplementedError("method not implemented")
    
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()
    
class MapEvaluator(Evaluator):
    
    def __init__(self, map_func=map):
        super(MapEvaluator, self).__init__()
        self.map_func = map_func
    
    def evaluate_all(self, jobs):
        return self.map_func(run_job, jobs)
    
class SubmitEvaluator(Evaluator):
    
    def __init__(self, submit_func):
        super(SubmitEvaluator, self).__init__()
        self.submit_func = submit_func
        
    def evaluate_all(self, jobs):
        futures = [self.submit_func(run_job, job) for job in jobs]
        return [f.result() for f in futures]

class ApplyEvaluator(Evaluator):
    
    def __init__(self, apply_func):
        super(ApplyEvaluator, self).__init__()
        self.apply_func = apply_func
        
    def evaluate_all(self, jobs):
        futures = [self.apply_func(run_job, [job]) for job in jobs]
        return [f.get() for f in futures]
    
class PoolEvaluator(MapEvaluator):
    
    def __init__(self, pool):
        super(PoolEvaluator, self).__init__(pool.map)
        self.pool = pool
        
    def close(self):
        self.pool.close()
        self.pool.join()
        
class MultiprocessingEvaluator(PoolEvaluator):
    
    def __init__(self, processes=None):
        super(MultiprocessingEvaluator, self).__init__(Pool(processes))
        