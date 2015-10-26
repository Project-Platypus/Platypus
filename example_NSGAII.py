from platypus.algorithms import NSGAII
from platypus.operators import TournamentSelector, PM, SBX, GAOperator
from platypus.problems import UF1
from platypus.core import Solution
from platypus.indicators import generational_distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

problem = UF1()

algorithm = NSGAII(problem,
                   population_size = 100,
                   selector = TournamentSelector(2),
                   variator = GAOperator(SBX(1.0), PM(1.0 / problem.nvars)))

sets = []

for _ in range(100):
    algorithm.run(100)
    sets.append(algorithm.result)
    
pf = []
with open("E:/Git/MOEAFramework/pf/UF1.dat", "r") as f:
    for line in f:
        solution = Solution(problem)
        solution.objectives[:] = map(float, line.split())
        pf.append(solution)
    
gd = generational_distance(pf)
print gd(sets)