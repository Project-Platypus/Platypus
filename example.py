from platypus import NSGAIII, DTLZ2, Recorder
from platypus import MainWindow
from PyQt5.QtGui import QApplication

# define the problem definition
problem = DTLZ2(3)

# instantiate the optimization algorithm
algorithm = NSGAIII(problem, divisions_outer=12)

# optimize the problem using 10,000 function evaluations
recorder = Recorder(save_result=True, save_all=False)

# GUI version
app = QApplication([])
window = MainWindow(nobjs=problem.nobjs, algorithm=algorithm, nfe=10000, recorder=recorder)
app.exec_()

# # non-GUI version
# algorithm.run(10000, recorder=recorder)

# plot the results using matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result])
plt.show()