# Copyright 2015 David Hadka
#
# This file is part of Platypus.
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
import math
import random
import operator
import functools
from platypus.core import Problem, Solution, EPSILON, evaluator
from platypus.types import Real
from abc import ABCMeta

################################################################################
# DTLZ Problems
################################################################################

class DTLZ1(Problem):
    
    def __init__(self, nobjs = 2):
        super(DTLZ1, self).__init__(nobjs+4, nobjs)
        self.types[:] = Real(0, 1)
        
    @evaluator
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = 100.0 * (k + sum([math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution.variables[self.nvars-k:]]))
        f = [0.5 * (1.0 + g)]*self.nobjs
        
        for i in range(self.nobjs):
            f[i] *= reduce(operator.mul,
                           [x for x in solution.variables[:self.nobjs-i-1]],
                           1)
            
            if i > 0:
                f[i] *= 1 - x[self.nobjs-i-1]
                
        solution.objectives[:] = f
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.nobjs-1] = [random.uniform(0.0, 1.0) for _ in range(self.nobjs-1)]
        solution.variables[self.nobjs-1:] = 0.5
        solution.evaluate()
        return solution

class DTLZ2(Problem):
    
    def __init__(self, nobjs = 2):
        super(DTLZ2, self).__init__(nobjs+9, nobjs)
        self.types[:] = Real(0, 1)
        
    @evaluator
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = sum([math.pow(x - 0.5, 2.0) for x in solution.variables[self.nvars-k:]])
        f = [1.0+g]*self.nobjs
        
        for i in range(self.nobjs):
            f[i] *= reduce(operator.mul,
                           [math.cos(0.5 * math.pi * x) for x in solution.variables[:self.nobjs-i-1]],
                           1)
            
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution.variables[self.nobjs-i-1])
        
        solution.objectives[:] = f
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.nobjs-1] = [random.uniform(0.0, 1.0) for _ in range(self.nobjs-1)]
        solution.variables[self.nobjs-1:] = 0.5
        solution.evaluate()
        return solution
        
class DTLZ3(Problem):
    
    def __init__(self, nobjs = 2):
        super(DTLZ3, self).__init__(nobjs+9, nobjs)
        self.types[:] = Real(0, 1)
        
    @evaluator
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = 100.0 * (k + sum([math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution.variables[self.nvars-k:]]))
        f = [1.0+g]*self.nobjs
        
        for i in range(self.nobjs):
            f[i] *= reduce(operator.mul,
                           [math.cos(0.5 * math.pi * x) for x in solution.variables[:self.nobjs-i-1]],
                           1)
            
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution.variables[self.nobjs-i-1])
        
        solution.objectives[:] = f
    
    def random(self):
        solution = Solution(self)
        solution.variables[:self.nobjs-1] = [random.uniform(0.0, 1.0) for _ in range(self.nobjs-1)]
        solution.variables[self.nobjs-1:] = 0.5
        solution.evaluate()
        return solution
        
class DTLZ4(Problem):
    
    def __init__(self, nobjs = 2, alpha = 100.0):
        super(DTLZ4, self).__init__(nobjs+9, nobjs)
        self.types[:] = Real(0, 1)
        self.alpha = alpha
        
    @evaluator
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = sum([math.pow(x - 0.5, 2.0) for x in solution.variables[self.nvars-k:]])
        f = [1.0+g]*self.nobjs
        
        for i in range(self.nobjs):
            f[i] *= reduce(operator.mul,
                           [math.cos(0.5 * math.pi * math.pow(x, self.alpha)) for x in solution.variables[:self.nobjs-i-1]],
                           1)
            
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * math.pow(solution.variables[self.nobjs-i-1], self.alpha))
        
        solution.objectives[:] = f
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.nobjs-1] = [random.uniform(0.0, 1.0) for _ in range(self.nobjs-1)]
        solution.variables[self.nobjs-1:] = 0.5
        solution.evaluate()
        return solution
        
        
class DTLZ7(Problem):
    
    def __init__(self, nobjs = 2):
        super(DTLZ7, self).__init__(nobjs+19, nobjs)
        self.types[:] = Real(0, 1)
        
    @evaluator
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = 1.0 + (9.0 * sum(solution.variables[self.nvars-k:])) / k
        h = self.nobjs - sum([x / (1.0 + g) * (1.0 + math.sin(3.0 * math.pi * x)) for x in solution.variables[:self.nobjs-1]])
        
        solution.objectives[:self.nobjs-1] = solution.variables[:self.nobjs-1]
        solution.objectives[-1] = (1.0 + g) * h
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.nobjs-1] = [random.uniform(0.0, 1.0) for _ in range(self.nobjs-1)]
        solution.variables[self.nobjs-1:] = 0.0
        solution.evaluate()
        return solution
        
        
################################################################################
# WFG Problems
################################################################################

def _normalize_z(z):
    return [z[i] / (2.0 * (i+1)) for i in range(len(z))]

def _correct_to_01(a):
    if a <= 0.0 and a >= -EPSILON:
        return 0.0
    elif a >= 1.0 and a <= 1.0 + EPSILON:
        return 1.0
    else:
        return a
    
def _vector_in_01(x):
    return all([a >= 0.0 and a <= 1.0 for a in x])

def _s_linear(y, A):
    return _correct_to_01(abs(y - A) / abs(math.floor(A - y) + A))

def _s_multi(y, A, B, C):
    tmp1 = abs(y - C) / (2.0 * (math.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * math.pi * (0.5 - tmp1)
    return _correct_to_01((1.0 + math.cos(tmp2) + 4.0 * B * math.pow(tmp1, 2.0)) / (B + 2.0))

def _s_decept(y, A, B, C):
    tmp1 = math.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = math.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    return _correct_to_01(1.0 + (abs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B))

def _b_flat(y, A, B, C):
    return _correct_to_01(A +
                          min(0.0, math.floor(y - B)) * A * (B - y) / B -
                          min(0.0, math.floor(C - y)) * (1.0 - A) * (y - C))

def _b_poly(y, alpha):
    return _correct_to_01(math.pow(y, alpha))

def _b_param(y, u, A, B, C):
    return _correct_to_01(math.pow(y, B + (C-B) * (A - (1.0 - 2.0*u) * abs(math.floor(0.5 - u) + A))))

def _subvector(v, head, tail):
    return [v[i] for i in range(head, tail)]

def _r_sum(y, w):
    numerator = sum([w[i]*y[i] for i in range(len(y))])
    denominator = sum([w[i] for i in range(len(y))])
    return _correct_to_01(numerator / denominator)

def _r_nonsep(y, A):
    numerator = sum([y[j] + sum([abs(y[j] - y[(j + k + 1) % len(y)]) for k in range(A-1)]) for j in range(len(y))])
    tmp = math.ceil(A / 2.0)
    denominator = len(y) * tmp * (1.0 + 2.0*A - 2.0*tmp) / A
    return _correct_to_01(numerator / denominator)

def _WFG1_t1(y, k):
    return y[:k] + map(functools.partial(_s_linear, A=0.35), y[k:])

def _WFG1_t2(y, k):
    return y[:k] + map(functools.partial(_b_flat, A=0.8, B=0.75, C=0.85), y[k:])

def _WFG1_t3(y):
    return map(functools.partial(_b_poly, alpha=0.02), y)

def _WFG1_t4(y, k, M):
    w = [2.0*(i+1) for i in range(len(y))]
    t = []
    
    for i in range(M-1):
        head = i * k / (M-1)
        tail = (i+1) * k / (M-1)
        y_sub = _subvector(y, head, tail)
        w_sub = _subvector(w, head, tail)
        t.append(_r_sum(y_sub, w_sub))

    y_sub = _subvector(y, k, len(y))
    w_sub = _subvector(w, k, len(y))
    t.append(_r_sum(y_sub, w_sub))    
    return t

def _WFG2_t2(y, k):
    l = len(y) - k
    t = y[:k]
    
    for i in range(k+1, k+(l/2)+1):
        head = k + 2 * (i - k) - 2
        tail = k + 2 * (i - k)
        t.append(_r_nonsep(_subvector(y, head, tail), 2))
        
    return t

def _WFG2_t3(y, k, M):
    w = [1.0]*len(y)
    t = []
    
    for i in range(M-1):
        head = i * k / (M-1)
        tail = (i+1) * k / (M-1)
        y_sub = _subvector(y, head, tail)
        w_sub = _subvector(w, head, tail)
        t.append(_r_sum(y_sub, w_sub))
        
    y_sub = _subvector(y, k, len(y))
    w_sub = _subvector(w, k, len(y))
    t.append(_r_sum(y_sub, w_sub))
    return t

def _WFG4_t1(y):
    return map(functools.partial(_s_multi, A=30, B=10, C=0.35), y)

def _WFG5_t1(y):
    return map(functools.partial(_s_decept, A=0.35, B=0.001, C=0.05), y)

def _WFG6_t2(y, k, M):
    t = []
    
    for i in range(M-1):
        head = i * k / (M-1)
        tail = (i+1) * k / (M-1)
        y_sub = _subvector(y, head, tail)
        t.append(_r_nonsep(y_sub, k / (M-1)))
        
    y_sub = _subvector(y, k, len(y))
    t.append(_r_nonsep(y_sub, len(y)-k))
    return t

def _WFG7_t1(y, k):
    w = [1.0]*len(y)
    t = []
    
    for i in range(k):
        y_sub = _subvector(y, i+1, len(y))
        w_sub = _subvector(w, i+1, len(y))
        u = _r_sum(y_sub, w_sub)
        t.append(_b_param(y[i], u, 0.98 / 49.98, 0.02, 50))
        
    for i in range(k, len(y)):
        t.append(y[i])
        
    return t

def _WFG8_t1(y, k):
    w = [1.0]*len(y)
    t = y[:k]
    
    for i in range(k, len(y)):
        y_sub = _subvector(y, 0, i)
        w_sub = _subvector(w, 0, i)
        u = _r_sum(y_sub, w_sub)
        t.append(_b_param(y[i], u, 0.98 / 49.98, 0.02, 50))
        
    return t

def _WFG9_t1(y):
    w = [1.0]*len(y)
    t = []
    
    for i in range(len(y)-1):
        y_sub = _subvector(y, i + 1, len(y))
        w_sub = _subvector(w, i + 1, len(y))
        u = _r_sum(y_sub, w_sub)
        t.append(_b_param(y[i], u, 0.98 / 49.98, 0.02, 50))
        
    t.append(y[-1])
    return t

def _WFG9_t2(y, k):
    return map(functools.partial(_s_decept, A=0.35, B=0.001, C=0.05), y[:k]) + map(functools.partial(_s_multi, A=30, B=95, C=0.35), y[k:])

def _create_A(M, degenerate):
    if degenerate:
        return [1.0 if i==0 else 0.0 for i in range(M-1)]
    else:
        return [1.0]*(M-1)
    
def _calculate_x(t_p, A):
    return [max(t_p[-1], A[i]) * (t_p[i] - 0.5) + 0.5 for i in range(len(t_p)-1)] + [t_p[-1]]

def _convex(x, m):
    result = reduce(operator.mul,
                    [1.0 - math.cos(x[i-1] * math.pi / 2.0) for i in range(1, len(x)-m+1)],
                    1.0)
    
    if m != 1:
        result *= 1.0 - math.sin(x[len(x)-m] * math.pi / 2.0)
        
    return _correct_to_01(result)

def _concave(x, m):
    result = reduce(operator.mul,
                    [math.sin(x[i-1] * math.pi / 2.0) for i in range(1, len(x)-m+1)],
                    1.0)
    
    if m != 1:
        result *= math.cos(x[len(x)-m] * math.pi / 2.0)
        
    return _correct_to_01(result)

def _linear(x, m):
    result = reduce(operator.mul, x[:len(x)-m], 1.0)
    
    if m != 1:
        result *= 1.0 - x[len(x)-m]
        
    return _correct_to_01(result)

def _mixed(x, A, alpha):
    tmp = 2.0 * A * math.pi
    return _correct_to_01(math.pow(1.0 - x[0] - math.cos(tmp * x[0] + math.pi / 2.0) / tmp, alpha))

def _disc(x, A, alpha, beta):
    tmp = A * math.pow(x[0], beta) * math.pi
    return _correct_to_01(1.0 - math.pow(x[0], alpha) * math.pow(math.cos(tmp), 2.0))

def _calculate_f(D, x, h, S):
    return [D * x[-1] + S[i]*h[i] for i in range(len(h))]

def _WFG_calculate_f(x, h):
    S = [m * 2.0 for m in range(1, len(h)+1)]
    return _calculate_f(1.0, x, h, S)

def _WFG1_shape(t_p):
    A = _create_A(len(t_p), False)
    x = _calculate_x(t_p, A)
    h = [_convex(x, m) for m in range(1, len(t_p))] + [_mixed(x, 5, 1.0)]
    return _WFG_calculate_f(x, h)

def _WFG2_shape(t_p):
    A = _create_A(len(t_p), False)
    x = _calculate_x(t_p, A)
    h = [_convex(x, m) for m in range(1, len(t_p))] + [_disc(x, 5, 1.0, 1.0)]
    return _WFG_calculate_f(x, h)

def _WFG3_shape(t_p):
    A = _create_A(len(t_p), True)
    x = _calculate_x(t_p, A)
    h = [_linear(x, m) for m in range(1, len(t_p)+1)]
    return _WFG_calculate_f(x, h)

def _WFG4_shape(t_p):
    A = _create_A(len(t_p), False)
    x = _calculate_x(t_p, A)
    h = [_concave(x, m) for m in range(1, len(t_p)+1)]
    return _WFG_calculate_f(x, h)

class WFG(Problem):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, k, l, m):
        super(WFG, self).__init__(k+l, m)
        self.k = k
        self.l = l
        self.m = m
        self.types[:] = [Real(0.0, 2.0*(i+1)) for i in range(k+l)]

class WFG1(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG1, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG1_t1(y, self.k)
        y = _WFG1_t2(y, self.k)
        y = _WFG1_t3(y)
        y = _WFG1_t4(y, self.k, self.m)
        y = _WFG1_shape(y)
        solution.objectives[:] = y
       
    def random(self):
        solution = Solution(self)
        solution.variables[:self.k] = [math.pow(random.uniform(0.0, 1.0), 50.0) for _ in range(self.k)]
        solution.variables[self.k:] = 0.35
        solution.variables[:] = [solution.variables[i] * 2.0 * (i+1) for i in range(self.nvars)]
        self.evaluate(solution)
        return solution

class WFG2(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG2, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG1_t1(y, self.k)
        y = _WFG2_t2(y, self.k)
        y = _WFG2_t3(y, self.k, self.m)
        y = _WFG2_shape(y)
        solution.objectives[:] = y
  
    def random(self):
        solution = Solution(self)
        solution.variables[:self.k] = [random.uniform(0.0, 1.0) for _ in range(self.k)]
        solution.variables[self.k:] = 0.35
        solution.variables[:] = [solution.variables[i] * 2.0 * (i+1) for i in range(self.nvars)]
        self.evaluate(solution)
        return solution
    
class WFG3(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG3, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG1_t1(y, self.k)
        y = _WFG2_t2(y, self.k)
        y = _WFG2_t3(y, self.k, self.m)
        y = _WFG3_shape(y)
        solution.objectives[:] = y
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.k] = [random.uniform(0.0, 1.0) for _ in range(self.k)]
        solution.variables[self.k:] = 0.35
        solution.variables[:] = [solution.variables[i] * 2.0 * (i+1) for i in range(self.nvars)]
        self.evaluate(solution)
        return solution
    
class WFG4(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG4, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG4_t1(y)
        y = _WFG2_t3(y, self.k, self.m)
        y = _WFG4_shape(y)
        solution.objectives[:] = y
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.k] = [random.uniform(0.0, 1.0) for _ in range(self.k)]
        solution.variables[self.k:] = 0.35
        solution.variables[:] = [solution.variables[i] * 2.0 * (i+1) for i in range(self.nvars)]
        self.evaluate(solution)
        return solution
    
class WFG5(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG5, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG5_t1(y)
        y = _WFG2_t3(y, self.k, self.m)
        y = _WFG4_shape(y)
        solution.objectives[:] = y
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.k] = [random.uniform(0.0, 1.0) for _ in range(self.k)]
        solution.variables[self.k:] = 0.35
        solution.variables[:] = [solution.variables[i] * 2.0 * (i+1) for i in range(self.nvars)]
        self.evaluate(solution)
        return solution
    
class WFG6(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG6, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG1_t1(y, self.k)
        y = _WFG6_t2(y, self.k, self.m)
        y = _WFG4_shape(y)
        solution.objectives[:] = y
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.k] = [random.uniform(0.0, 1.0) for _ in range(self.k)]
        solution.variables[self.k:] = 0.35
        solution.variables[:] = [solution.variables[i] * 2.0 * (i+1) for i in range(self.nvars)]
        self.evaluate(solution)
        return solution
    
class WFG7(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG7, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG7_t1(y, self.k)
        y = _WFG1_t1(y, self.k)
        y = _WFG2_t3(y, self.k, self.m)
        y = _WFG4_shape(y)
        solution.objectives[:] = y
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.k] = [random.uniform(0.0, 1.0) for _ in range(self.k)]
        solution.variables[self.k:] = 0.35
        solution.variables[:] = [solution.variables[i] * 2.0 * (i+1) for i in range(self.nvars)]
        self.evaluate(solution)
        return solution
    
class WFG8(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG8, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG8_t1(y, self.k)
        y = _WFG1_t1(y, self.k)
        y = _WFG2_t3(y, self.k, self.m)
        y = _WFG4_shape(y)
        solution.objectives[:] = y
        
    def random(self):
        result = [random.uniform(0.0, 1.0) for _ in range(self.k)] + [0.0]*self.l

        for i in range(self.k, self.nvars):
            w = [1.0]*(self.nvars)
            u = _r_sum(result, w)
            tmp1 = abs(math.floor(0.5 - u) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1.0 - 2.0*u) * tmp1)
            result[i] = math.pow(0.35, math.pow(tmp2, -1.0))
            
        result = [result[i] * 2.0 * (i+1) for i in range(self.nvars)]
        
        solution = Solution(self)
        solution.variables[:] = result
        self.evaluate(solution)
        return solution
    
class WFG9(WFG):
    
    def __init__(self, nobjs = 2):
        super(WFG9, self).__init__(nobjs-1, 10, nobjs)
        
    @evaluator
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG9_t1(y)
        y = _WFG9_t2(y, self.k)
        y = _WFG6_t2(y, self.k, self.m)
        y = _WFG4_shape(y)
        solution.objectives[:] = y
        
    def random(self):
        result = [random.uniform(0.0, 1.0) for _ in range(self.k)] + [0.0]*(self.l-1) + [0.35]

        for i in range(self.nvars-2, self.k-1, -1):
            result_sub = result[i+1:self.nvars]
            w = [1.0]*len(result_sub)
            tmp1 = _r_sum(result_sub, w)
            result[i] = math.pow(0.35, math.pow(0.02 + 1.96 * tmp1, -1.0))

        result = [result[i] * 2.0 * (i+1) for i in range(self.nvars)]
        
        solution = Solution(self)
        solution.variables[:] = result
        self.evaluate(solution)
        return solution
    
################################################################################
# CEC 2009 Problems
################################################################################

class UF1(Problem):
    
    def __init__(self, nvars = 30):
        super(UF1, self).__init__(nvars, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-1, 1)
    
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        
        for j in range(2, self.nvars+1):
            yj = x[j-1] - math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
            
            if j % 2 == 1:
                sum1 += yj**2
                count1 += 1
            else:
                sum2 += yj**2
                count2 += 1
                
        f1 = x[0] + 2.0 * sum1 / count1
        f2 = 1.0 - math.sqrt(x[0]) + 2.0 * sum2 / count2
        solution.objectives[:] = [f1, f2]

class UF2(Problem):
    
    def __init__(self, nvars = 30):
        super(UF2, self).__init__(nvars, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-1, 1)
    
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        
        for j in range(2, self.nvars+1):
            if j % 2 == 1:
                yj = x[j-1] - 0.3*x[0]*(x[0] * math.cos(24.0*math.pi*x[0] + 4.0*j*math.pi/self.nvars) + 2.0)*math.cos(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                sum1 += yj**2
                count1 += 1
            else:
                yj = x[j-1] - 0.3*x[0]*(x[0] * math.cos(24.0*math.pi*x[0] + 4.0*j*math.pi/self.nvars) + 2.0)*math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                sum2 += yj**2
                count2 += 1
                
        f1 = x[0] + 2.0 * sum1 / count1
        f2 = 1.0 - math.sqrt(x[0]) + 2.0 * sum2 / count2
        solution.objectives[:] = [f1, f2]

class UF3(Problem):
    
    def __init__(self, nvars = 30):
        super(UF3, self).__init__(nvars, 2)
        self.types[:] = Real(0, 1)
    
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        prod1 = 1.0
        prod2 = 1.0
        
        for j in range(2, self.nvars+1):
            yj = x[j-1] - math.pow(x[0], 0.5*(1.0 + 3.0*(j - 2.0) / (self.nvars - 2.0)))
            pj = math.cos(20.0*yj*math.pi/math.sqrt(j))
            
            if j % 2 == 1:
                sum1 += yj**2
                prod1 *= pj
                count1 += 1
            else:
                sum2 += yj**2
                prod2 *= pj
                count2 += 1
                
        f1 = x[0] + 2.0 * (4.0*sum1 - 2.0*prod1 + 2.0) / count1
        f2 = 1.0 - math.sqrt(x[0]) + 2.0 * (4.0*sum2 - 2.0*prod2 + 2.0) / count2
        solution.objectives[:] = [f1, f2]

class UF4(Problem):
    
    def __init__(self, nvars = 30):
        super(UF4, self).__init__(nvars, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-2, 2)
    
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        
        for j in range(2, self.nvars+1):
            yj = x[j-1] - math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
            hj = abs(yj) / (1.0 + math.exp(2.0*abs(yj)))
            
            if j % 2 == 1:
                sum1 += hj
                count1 += 1
            else:
                sum2 += hj
                count2 += 1
                
        f1 = x[0] + 2.0*sum1/count1
        f2 = 1.0 - x[0]**2 + 2.0*sum2/count2
        solution.objectives[:] = [f1, f2]

class UF5(Problem):
    
    def __init__(self, nvars = 30):
        super(UF5, self).__init__(nvars, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-1, 1)
    
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        N = 10.0
        E = 0.1
        
        for j in range(2, self.nvars+1):
            yj = x[j-1] - math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
            hj = 2.0*yj**2 - math.cos(4.0*math.pi*yj) + 1.0
            
            if j % 2 == 1:
                sum1 += hj
                count1 += 1
            else:
                sum2 += hj
                count2 += 1
        
        hj = (0.5/N + E) * abs(math.sin(2.0*N*math.pi*x[0]))
        f1 = x[0] + hj + 2.0*sum1/count1
        f2 = 1.0 - x[0] + hj + 2.0*sum2/count2
        solution.objectives[:] = [f1, f2]

class CF1(Problem):
    
    def __init__(self, nvars = 10):
        super(CF1, self).__init__(nvars, 2, 1)
        self.types[:] = Real(0, 1)
        self.constraints[:] = ">=0"

    @evaluator        
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        N = 10.0
        a = 1.0
        
        for j in range(2, self.nvars+1):
            yj = x[j-1] - math.pow(x[0], 0.5 * (1.0 + 3.0 * (j - 2.0) / (self.nvars - 2.0)))
            
            if j % 2 == 1:
                sum1 += yj * yj
                count1 += 1
            else:
                sum2 += yj * yj
                count2 += 1
                
        f1 = x[0] + 2.0 * sum1 / count1
        f2 = 1.0 - x[0] + 2.0 * sum2 / count2
        solution.objectives[:] = [f1, f2]
        solution.constraints[0] = f1 + f2 - a * abs(math.sin(N * math.pi * (f1 - f2 + 1.0))) - 1.0
        
class CF2(Problem):
    
    def __init__(self, nvars = 10):
        super(CF2, self).__init__(nvars, 2, 1)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-1, 1)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        N = 2.0
        a = 1.0
        
        for j in range(2, self.nvars+1):
            if j % 2 == 1:
                yj = x[j-1] - math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                sum1 += yj * yj
                count1 += 1
            else:
                yj = x[j-1] - math.cos(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                sum2 += yj * yj
                count2 += 1
                
        f1 = x[0] + 2.0 * sum1 / count1
        f2 = 1.0 - math.sqrt(x[0]) + 2.0 * sum2 / count2
        t = f2 + math.sqrt(f1) - a * math.sin(N * math.pi * (math.sqrt(f1) - f2 + 1.0)) - 1.0
        solution.objectives[:] = [f1, f2]
        solution.constraints[0] = (1 if t >= 0 else -1) * abs(t) / (1.0 + math.exp(4.0 * abs(t)))
        
class CF3(Problem):
    
    def __init__(self, nvars = 10):
        super(CF3, self).__init__(nvars, 2, 1)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        prod1 = 1.0
        prod2 = 1.0
        N = 2.0
        a = 1.0
        
        for j in range(2, self.nvars+1):
            yj = x[j-1] - math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
            pj = math.cos(20.0 * yj * math.pi / math.sqrt(j))
            
            if j % 2 == 1:
                sum1 += yj * yj
                prod1 *= pj
                count1 += 1
            else:
                sum2 += yj * yj
                prod2 *= pj
                count2 += 1
                
        f1 = x[0] + 2.0 * (4.0 * sum1 - 2.0 * prod1 + 2.0) / count1
        f2 = 1.0 - x[0]**2 + 2.0 * (4.0 * sum2 - 2.0 * prod2 + 2.0) / count2
        solution.objectives[:] = [f1, f2]
        solution.constraints[0] = f2 + f1**2 - a * math.sin(N * math.pi * (f1**2 - f2 + 1.0)) - 1.0
        
class CF4(Problem):
    
    def __init__(self, nvars = 10):
        super(CF4, self).__init__(nvars, 2, 1)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        sum1 = 0.0
        sum2 = 0.0
        
        for j in range(2, self.nvars+1):
            yj = x[j-1] - math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
            
            if j % 2 == 1:
                sum1 += yj**2
            else:
                if j == 2:
                    sum2 += abs(yj) if yj < 1.5 - 0.75 * math.sqrt(2.0) else 0.125 + (yj - 1)**2
                else:
                    sum2 += yj**2
                                    
        f1 = x[0] + sum1
        f2 = 1.0 - x[0] + sum2
        t = x[1] - math.sin(6.0*x[0]*math.pi + 2.0*math.pi/self.nvars) - 0.5*x[0] + 0.25
        solution.objectives[:] = [f1, f2]
        solution.constraints[0] = (1 if t >= 0 else -1) * abs(t) / (1.0 + math.exp(4.0 * abs(t)))
        
class CF5(Problem):
    
    def __init__(self, nvars = 10):
        super(CF5, self).__init__(nvars, 2, 1)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        sum1 = 0.0
        sum2 = 0.0
        
        for j in range(2, self.nvars+1):            
            if j % 2 == 1:
                yj = x[j-1] - 0.8*x[0]*math.cos(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                sum1 += 2.0*yj**2 - math.cos(4.0*math.pi*yj) + 1.0
            else:
                yj = x[j-1] - 0.8*x[0]*math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                
                if j == 2:
                    sum2 += abs(yj) if yj < 1.5 - 0.75*math.sqrt(2.0) else 0.125 + (yj - 1)**2
                else:
                    sum2 += 2.0*yj**2 - math.cos(4.0*math.pi*yj) + 1.0
                                    
        f1 = x[0] + sum1
        f2 = 1.0 - x[0] + sum2
        solution.objectives[:] = [f1, f2]
        solution.constraints[0] = x[1] - 0.8*x[0]*math.sin(6.0*x[0]*math.pi + 2.0*math.pi/self.nvars) - 0.5*x[0] + 0.25
        
class CF6(Problem):
    
    def __init__(self, nvars = 10):
        super(CF6, self).__init__(nvars, 2, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        sum1 = 0.0
        sum2 = 0.0
        
        for j in range(2, self.nvars+1):            
            if j % 2 == 1:
                yj = x[j-1] - 0.8*x[0]*math.cos(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                sum1 += yj**2
            else:
                yj = x[j-1] - 0.8*x[0]*math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                sum2 += yj**2
                                    
        f1 = x[0] + sum1
        f2 = (1.0 - x[0])**2 + sum2
        c1 = x[1] - 0.8*x[0]*math.sin(6.0*x[0]*math.pi + 2.0*math.pi/self.nvars) - \
                (1 if (x[0]-0.5)*(1.0-x[0]) >= 0 else -1) * math.sqrt(abs((x[0]-0.5)*(1.0-x[0])))
        c2 = x[3] - 0.8*x[0]*math.sin(6.0*x[0]*math.pi + 4.0*math.pi/self.nvars) - \
                (1 if 0.25 * math.sqrt(1.0-x[0]) - 0.5*(1.0-x[0]) >= 0 else -1)*math.sqrt(abs(0.25 * math.sqrt(1-x[0]) - 0.5*(1.0-x[0])))
        solution.objectives[:] = [f1, f2]
        solution.constraints[:] = [c1, c2]

class CF7(Problem):
    
    def __init__(self, nvars = 10):
        super(CF7, self).__init__(nvars, 2, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        sum1 = 0.0
        sum2 = 0.0
        
        for j in range(2, self.nvars+1):            
            if j % 2 == 1:
                yj = x[j-1] - math.cos(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                sum1 += 2.0*yj**2 - math.cos(4.0*math.pi*yj) + 1.0
            else:
                yj = x[j-1] - math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
                
                if j == 2 or j == 4:
                    sum2 += yj**2
                else:
                    sum2 += 2.0*yj**2 - math.cos(4.0*math.pi*yj) + 1.0
                                    
        f1 = x[0] + sum1
        f2 = (1.0 - x[0])**2 + sum2
        c1 = x[1] - math.sin(6.0*x[0]*math.pi + 2.0*math.pi/self.nvars) - \
                (1 if (x[0]-0.5)*(1.0-x[0]) >= 0 else -1) * math.sqrt(abs((x[0]-0.5)*(1.0-x[0])))
        c2 = x[3] - math.sin(6.0*x[0]*math.pi + 4.0*math.pi/self.nvars) - \
                (1 if 0.25 * math.sqrt(1.0-x[0]) - 0.5*(1.0-x[0]) >= 0 else -1)*math.sqrt(abs(0.25 * math.sqrt(1-x[0]) - 0.5*(1.0-x[0])))
        solution.objectives[:] = [f1, f2]
        solution.constraints[:] = [c1, c2]
                                            
class CF8(Problem):
    
    def __init__(self, nvars = 10):
        super(CF8, self).__init__(nvars, 3, 1)
        self.types[0:1] = Real(0, 1)
        self.types[2:] = Real(-4, 4)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        count3 = 0
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        N = 2.0
        a = 4.0
        
        for j in range(3, self.nvars+1):
            yj = x[j-1] - 2.0*x[1]*math.sin(2.0*math.pi*x[0] + j*math.pi/self.nvars)
                        
            if j % 3 == 1:
                sum1 += yj**2
                count1 += 1
            elif j % 3 == 2:
                sum2 += yj**2
                count2 += 1
            else:
                sum3 += yj**2
                count3 += 1
                                    
        f1 = math.cos(0.5*math.pi*x[0]) * math.cos(0.5*math.pi*x[1]) + 2.0*sum1/count1
        f2 = math.cos(0.5*math.pi*x[0]) * math.sin(0.5*math.pi*x[1]) + 2.0*sum2/count2
        f3 = math.sin(0.5*math.pi*x[0]) + 2.0*sum3/count3
        c1 = (f1**2 + f2**2) / (1.0 - f3**2) - a*abs(math.sin(N*math.pi*((f1**2 - f2**2) / (1.0 - f3**2) + 1.0))) - 1.0
        solution.objectives[:] = [f1, f2]
        solution.constraints[:] = [c1]
                 
class CF9(Problem):
    
    def __init__(self, nvars = 10):
        super(CF9, self).__init__(nvars, 3, 1)
        self.types[0:1] = Real(0, 1)
        self.types[2:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        count3 = 0
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        N = 2.0
        a = 3.0
        
        for j in range(3, self.nvars+1):
            yj = x[j-1] - 2.0*x[1]*math.sin(2.0*math.pi*x[0] + j*math.pi/self.nvars)
                        
            if j % 3 == 1:
                sum1 += yj**2
                count1 += 1
            elif j % 3 == 2:
                sum2 += yj**2
                count2 += 1
            else:
                sum3 += yj**2
                count3 += 1
                                    
        f1 = math.cos(0.5*math.pi*x[0]) * math.cos(0.5*math.pi*x[1]) + 2.0*sum1/count1
        f2 = math.cos(0.5*math.pi*x[0]) * math.sin(0.5*math.pi*x[1]) + 2.0*sum2/count2
        f3 = math.sin(0.5*math.pi*x[0]) + 2.0*sum3/count3
        c1 = (f1**2 + f2**2) / (1.0 - f3**2) - a*math.sin(N*math.pi*((f1**2 - f2**2) / (1.0 - f3**2) + 1.0)) - 1.0
        solution.objectives[:] = [f1, f2]
        solution.constraints[:] = [c1]

class CF10(Problem):
    
    def __init__(self, nvars = 10):
        super(CF10, self).__init__(nvars, 3, 1)
        self.types[0:1] = Real(0, 1)
        self.types[2:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
    @evaluator
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        count3 = 0
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        N = 2.0
        a = 1.0
        
        for j in range(3, self.nvars+1):
            yj = x[j-1] - 2.0*x[1]*math.sin(2.0*math.pi*x[0] + j*math.pi/self.nvars)
            hj = 4.0*yj**2 - math.cos(8.0*math.pi*yj) + 1.0
                        
            if j % 3 == 1:
                sum1 += hj
                count1 += 1
            elif j % 3 == 2:
                sum2 += hj
                count2 += 1
            else:
                sum3 += hj
                count3 += 1
                                    
        f1 = math.cos(0.5*math.pi*x[0]) * math.cos(0.5*math.pi*x[1]) + 2.0*sum1/count1
        f2 = math.cos(0.5*math.pi*x[0]) * math.sin(0.5*math.pi*x[1]) + 2.0*sum2/count2
        f3 = math.sin(0.5*math.pi*x[0]) + 2.0*sum3/count3
        c1 = (f1**2 + f2**2) / (1.0 - f3**2) - a*math.sin(N*math.pi*((f1**2 - f2**2) / (1.0 - f3**2) + 1.0)) - 1.0
        solution.objectives[:] = [f1, f2]
        solution.constraints[:] = [c1]
                                  