from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time
import traceback, sys
from threading import Thread
import pyqtgraph as pg 


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
        finished
            No data
        error
            `tuple` (exctype, value, traceback.format_exc() )
        result
            `object` data returned from processing, anything
        progress
            `int` indicating % progress
    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class MainWindow(QMainWindow):

    def __init__(self, nobjs, algorithm, nfe, recorder, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.nobjs = nobjs
        self.algorithm = algorithm
        self.nfe = nfe
        self.recorder = recorder

        self._init_UI()

        layout = QVBoxLayout()
        layout.addWidget(self.clock)
        layout.addWidget(self.l)
        layout.addWidget(self.win)

        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)
        self.show()

        self._run_qtimer()  # starts timing application

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Sets observee
        self._subject = None
        self.recorder.attach(self)

        thread = Thread(target=self._threaded_algorithm)
        thread.start()

    def update(self, population):
        # print('result recv with {} items'.format(len(population)))
        worker = Worker(self._execute_this_plot, population)
        worker.signals.result.connect(self._print_output)
        worker.signals.finished.connect(self._thread_complete)
        worker.signals.progress.connect(self._progress_fn)

        # Execute
        self.threadpool.start(worker)

    def _threaded_algorithm(self):
        for i in range(1, 0, -1):
            self.l.setText('Algorithm starts at %d' % i)
            time.sleep(1)
        self.l.setText("Running")
        self.algorithm.run(self.nfe, recorder=self.recorder)

    def _progress_fn(self, n):
        # print("%d%% done" % n)
        pass

    def _print_output(self, s):
        # print(s)
        pass

    def _thread_complete(self):
        # print("THREAD COMPLETE!")
        pass

    def _execute_this_plot(self, population, progress_callback):
        print('no of pop={}'.format(len(population)))
        for i in range(self.nobjs):
            objs = [s.objectives[i] for s in population] 
            avg_obj = float(sum(objs)) / float(len(objs))
            min_obj = min(objs)
            self.curves_o[i].setData(self.Xms_o[i])
            self.Xms_b[i].append(min_obj)
            self.curves_b[i].setData(self.Xms_b[i])
            self.Xms_o[i].append(avg_obj)
            progress_callback.emit((i+1)*100/self.nobjs)

    def _init_UI(self):
        self._init_clock()
        self._init_label()
        self._init_plot_window()

    def _run_qtimer(self):
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._recurring_timer)
        self.timer.start()

    def _init_clock(self):
        self.counter = 0
        self.clock = QLabel("Time elapsed: %ds" % self.counter)

    def _init_label(self):
        self.l = QLabel("Start")

    def _init_plot_window(self):
        self.win = pg.GraphicsWindow()
        self.win.resize(1000, 800)

        # Average objectives over generation
        self.ps_o     = [None] * self.nobjs
        self.curves_o = [None] * self.nobjs
        self.Xms_o    = [[] for i in range(self.nobjs)]

        # Initialise average objective plots
        for i in range(self.nobjs):
            p = self.win.addPlot(title="Average obj[{}]".format(i))  # creates a PlotItem
            self.ps_o[i] = p

        # Line plots for average objectives
        for i, p in zip(range(self.nobjs), self.ps_o):
            self.curves_o[i] = p.plot(pen=(255, 0, 0))
        self.win.nextRow()

        # Best objectives
        self.ps_b     = [None] * self.nobjs
        self.curves_b = [None] * self.nobjs
        self.Xms_b    = [[] for i in range(self.nobjs)]

        # Initialise best objective plots
        for i in range(self.nobjs):
            p = self.win.addPlot(title="Best (min) obj[{}]".format(i))  # creates a PlotItem
            self.ps_b[i] = p

        # Scatter plots for best objectives        
        for i, p in zip(range(self.nobjs), self.ps_b):
            self.curves_b[i] = p.plot([], pen=None, symbolBrush=(0,255,0), symbolSize=5, symbolPen=None)

    def _recurring_timer(self):
        self.counter +=1
        self.clock.setText("Time elapsed: %ds" % self.counter)



class RouteMainWindow(QMainWindow):

    def __init__(self, nobjs, algorithm, nfe, recorder, *args, **kwargs):
        super(RouteMainWindow, self).__init__(*args, **kwargs)
        self.nobjs = nobjs
        self.algorithm = algorithm
        self.nfe = nfe
        self.recorder = recorder

        self._init_UI()

        layout = QVBoxLayout()
        layout.addWidget(self.clock)
        layout.addWidget(self.l)
        layout.addWidget(self.win)

        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)
        self.show()

        self._run_qtimer()  # starts timing application

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Sets observee
        self._subject = None
        self.recorder.attach(self)

        thread = Thread(target=self._threaded_algorithm)
        thread.start()

    def update(self, population):
        # print('result recv with {} items'.format(len(population)))
        worker = Worker(self._execute_this_plot, population)
        worker.signals.result.connect(self._print_output)
        worker.signals.finished.connect(self._thread_complete)
        worker.signals.progress.connect(self._progress_fn)

        # Execute
        self.threadpool.start(worker)

    def _threaded_algorithm(self):
        for i in range(1, 0, -1):
            self.l.setText('Algorithm starts at %d' % i)
            time.sleep(1)
        self.l.setText("Running")
        self.algorithm.run(self.nfe, recorder=self.recorder)

    def _progress_fn(self, n):
        # print("%d%% done" % n)
        pass

    def _print_output(self, s):
        # print(s)
        pass

    def _thread_complete(self):
        # print("THREAD COMPLETE!")
        pass

    def _execute_this_plot(self, population, progress_callback):
        import numpy as np
        for i in range(self.nobjs):
            objs = [s.objectives[i] for s in population]
            min_obj_idx = np.argmin(objs)
            candidate = population[min_obj_idx]
            route = candidate.metadata['coords_route']
            self.ps_o[i].plot(route[0], route[1], clear=True)
            progress_callback.emit((i+1)*100/self.nobjs)

    def _init_UI(self):
        self._init_clock()
        self._init_label()
        self._init_plot_window()

    def _run_qtimer(self):
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._recurring_timer)
        self.timer.start()

    def _init_clock(self):
        self.counter = 0
        self.clock = QLabel("Time elapsed: %ds" % self.counter)

    def _init_label(self):
        self.l = QLabel("Start")

    def _init_plot_window(self):
        self.win = pg.GraphicsWindow()
        self.win.resize(1000, 800)

        # Hyperline over generations
        self.ps_o     = [None] * self.nobjs
        self.curves_o = [None] * self.nobjs

        # Initialise hyperline plots
        for i in range(self.nobjs):
            p = self.win.addPlot(title="Hyperline")  # creates a PlotItem
            self.ps_o[i] = p

        # scatter plots for hyperline
        for i, p in zip(range(self.nobjs), self.ps_o):
            self.curves_o[i] = p.plot([], pen=None, symbolBrush=(0,255,0), symbolSize=5, symbolPen=None)

    def _recurring_timer(self):
        self.counter +=1
        self.clock.setText("Time elapsed: %ds" % self.counter)


