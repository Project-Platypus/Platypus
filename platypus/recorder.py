"""Platypus result module.
Currently saves everything as a pickle object."""

import pickle
import datetime


class Recorder:
    """A Recorder object which saves itself when commit() is called.
    The current properties of the Algorithm class which could be accessed through
    `recorder.<name-of-property>` are:
        problem : <Problem>
        population_size : int
        evaluator : <Evaluator>
        variator : <Variator>
        generator : <Generator>
        selector : <Selector>
        nfe : int
        result : array of <Solution>
        all_res : dict of results
    """

    def __init__(self, save_all=False):
        """When save_all is set to True, save all generation results"""
        self.save_all = save_all
        self._problem = None
        self._population_size = None
        # Consider saving a string repr of these objects
        self._evaluator = None 
        self._variator = None
        self._generator = None
        self._selector = None
        self._nfe = None
        self._result = None
        self._all_res = None

    def add_generation_result(self, result):
        if not self._all_res:
            self._all_res = {}
            self._ngen = 0
        self.all_res[self._ngen] = result
        self._ngen += 1

    def commit(self, algorithm):
        self._problem = algorithm.problem
        try:
            self._population_size = algorithm.population_size
        except NameError:
            pass
        self._evaluator = algorithm.evaluator
        try:
            self._variator = algorithm.variator
        except NameError:
            pass
        try:
            self._generator = algorithm.generator
        except NameError:
            pass
        try:
            self._selector = algorithm.selector
        except NameError:
            pass
        self._nfe = algorithm.nfe
        self._result = algorithm.result

        self.save_to_file()

    def save_to_file(self):
        # Append timestamp to filename
        filename = f'{datetime.datetime.now():%Y%m%d_%H%M%S}'

        if self._population_size:
            filename += "_npop{}".format(self._population_size)
        
        if self._nfe:
            filename += "_nfe{}".format(self._nfe)

        if self._all_res:
            """Saves results from all generations"""
            filename += "_all"
        
        filehandler = open(filename, 'wb')
        pickle.dump(self, filehandler)

    @property
    def problem(self):
        return self._problem
    @property
    def population_size(self):
        return self._population_size
    @property
    def evaluator(self):
        return self._evaluator
    @property
    def variator(self):
        return self._variator
    @property
    def generator(self):
        return self._generator
    @property
    def selector(self):
        return self._selector
    @property
    def nfe(self):
        return self._nfe
    @property
    def result(self):
        return self._result
    @property
    def all_res(self):
        return self._all_res
