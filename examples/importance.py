import numpy as np
import matplotlib.pyplot as plt
from platypus import NSGAII, Problem, Real, Integer, CompoundOperator, SBX, PM, Type, HUX, BitFlip

"""
NOTE:
関連する要素が結果に与える影響を把握したい
- 探索範囲と収束の関係性
    - 数値範囲比較, (-10, 10)/(0, 2)
    - 実数、整数比較
- 世代数と集団のサイズが収束に与える影響
    - gen 10, 100, 1000 / pop 100
    - pop 10, 100, 1000 / gen 1000
"""



def schaffer(x):
    return [x[0]**2, (x[0]-2)**2]

# 1 - 1
POP = 100
MIN = -10
MAX = 10
GEN = 1000
problem = Problem(1, 2)
problem.types[:] = Real(MIN, MAX)
problem.function = schaffer

algorithm = NSGAII(problem, population_size=POP)
algorithm.run(GEN)

fig = plt.figure()
fig.suptitle("GEN:%d_POP:%d_MIN-MAX:%d-%d" %(GEN, POP, MIN, MAX))

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
for i, result in enumerate(algorithm.result):
    ax1.scatter(i, result.variables, color='k')
    ax1.set_title('Variables')
    ax2.scatter(result.objectives[0], result.objectives[1], color='k')
    ax2.set_title('Objectives')
plt.show()

# 1 - 2
POP = 100
MIN = -10
MAX = 10
GEN = 1000

import random
class Int(Type):
    def __init__(self, min_value, max_value):
        super(Int).__init__()
        self.min_value = min_value
        self.max_value = max_value
    
    def rand(self):
        return random.randint(self.min_value, self.max_value)

problem = Problem(1, 2)
problem.types[:] = Integer(MIN, MAX)
problem.function = schaffer

algorithm = NSGAII(problem, population_size=POP)
algorithm.run(GEN)

fig = plt.figure()
fig.suptitle("GEN:%d_POP:%d_MIN-MAX:%d-%d" %(GEN, POP, MIN, MAX))

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
for i, result in enumerate(algorithm.result):
    a = (np.array(result.variables[0])).astype(int)
    v = int(''.join(map(str, a)), base=2)
    print(v, a)
    ax1.scatter(i, v, color='k')
    ax1.set_title('Variables')
    ax2.scatter(result.objectives[0], result.objectives[1], color='k')
    ax2.set_title('Objectives')
plt.show()
