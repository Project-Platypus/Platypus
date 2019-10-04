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

import math
import random
import operator
import functools
from .core import Problem, Solution, EPSILON
from .types import Real, Binary
from abc import ABCMeta

################################################################################
# DTLZ Problems
################################################################################

class DTLZ1(Problem):
    
    def __init__(self, nobjs = 2):
        super(DTLZ1, self).__init__(nobjs+4, nobjs)
        self.types[:] = Real(0, 1)
        
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = 100.0 * (k + sum([math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution.variables[self.nvars-k:]]))
        f = [0.5 * (1.0 + g)]*self.nobjs
        
        for i in range(self.nobjs):
            f[i] *= functools.reduce(operator.mul,
                           [x for x in solution.variables[:self.nobjs-i-1]],
                           1)
            
            if i > 0:
                f[i] *= 1 - solution.variables[self.nobjs-i-1]
                
        solution.objectives[:] = f
        
    def random(self):
        solution = Solution(self)
        solution.variables[:self.nobjs-1] = [random.uniform(0.0, 1.0) for _ in range(self.nobjs-1)]
        solution.variables[self.nobjs-1:] = 0.5
        solution.evaluate()
        return solution

class DTLZ2(Problem):
    
    def __init__(self, nobjs = 2, nvars=None):
        super(DTLZ2, self).__init__(nobjs+9 if nvars is None else nvars, nobjs)
        self.types[:] = Real(0, 1)
        
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = sum([math.pow(x - 0.5, 2.0) for x in solution.variables[self.nvars-k:]])
        f = [1.0+g]*self.nobjs

        for i in range(self.nobjs):
            f[i] *= functools.reduce(operator.mul,
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
    
    def __init__(self, nobjs = 2, nvars=None):
        super(DTLZ3, self).__init__(nobjs+9 if nvars is None else nvars, nobjs)
        self.types[:] = Real(0, 1)
        
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = 100.0 * (k + sum([math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution.variables[self.nvars-k:]]))
        f = [1.0+g]*self.nobjs
        
        for i in range(self.nobjs):
            f[i] *= functools.reduce(operator.mul,
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
        
    def evaluate(self, solution):
        k = self.nvars - self.nobjs + 1
        g = sum([math.pow(x - 0.5, 2.0) for x in solution.variables[self.nvars-k:]])
        f = [1.0+g]*self.nobjs
        
        for i in range(self.nobjs):
            f[i] *= functools.reduce(operator.mul,
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
    return y[:k] + list(map(functools.partial(_s_linear, A=0.35), y[k:]))

def _WFG1_t2(y, k):
    return y[:k] + list(map(functools.partial(_b_flat, A=0.8, B=0.75, C=0.85), y[k:]))

def _WFG1_t3(y):
    return list(map(functools.partial(_b_poly, alpha=0.02), y))

def _WFG1_t4(y, k, M):
    w = [2.0*(i+1) for i in range(len(y))]
    t = []
    
    for i in range(M-1):
        head = i * k // (M-1)
        tail = (i+1) * k // (M-1)
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
    
    for i in range(k+1, k+(l//2)+1):
        head = k + 2 * (i - k) - 2
        tail = k + 2 * (i - k)
        t.append(_r_nonsep(_subvector(y, head, tail), 2))
        
    return t

def _WFG2_t3(y, k, M):
    w = [1.0]*len(y)
    t = []
    
    for i in range(M-1):
        head = i * k // (M-1)
        tail = (i+1) * k // (M-1)
        y_sub = _subvector(y, head, tail)
        w_sub = _subvector(w, head, tail)
        t.append(_r_sum(y_sub, w_sub))
        
    y_sub = _subvector(y, k, len(y))
    w_sub = _subvector(w, k, len(y))
    t.append(_r_sum(y_sub, w_sub))
    return t

def _WFG4_t1(y):
    return list(map(functools.partial(_s_multi, A=30, B=10, C=0.35), y))

def _WFG5_t1(y):
    return list(map(functools.partial(_s_decept, A=0.35, B=0.001, C=0.05), y))

def _WFG6_t2(y, k, M):
    t = []
    
    for i in range(M-1):
        head = i * k // (M-1)
        tail = (i+1) * k // (M-1)
        y_sub = _subvector(y, head, tail)
        t.append(_r_nonsep(y_sub, k // (M-1)))
        
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
    return list(map(functools.partial(_s_decept, A=0.35, B=0.001, C=0.05), y[:k])) + list(map(functools.partial(_s_multi, A=30, B=95, C=0.35), y[k:]))

def _create_A(M, degenerate):
    if degenerate:
        return [1.0 if i==0 else 0.0 for i in range(M-1)]
    else:
        return [1.0]*(M-1)
    
def _calculate_x(t_p, A):
    return [max(t_p[-1], A[i]) * (t_p[i] - 0.5) + 0.5 for i in range(len(t_p)-1)] + [t_p[-1]]

def _convex(x, m):
    result = functools.reduce(operator.mul,
                    [1.0 - math.cos(x[i-1] * math.pi / 2.0) for i in range(1, len(x)-m+1)],
                    1.0)
    
    if m != 1:
        result *= 1.0 - math.sin(x[len(x)-m] * math.pi / 2.0)
        
    return _correct_to_01(result)

def _concave(x, m):
    result = functools.reduce(operator.mul,
                    [math.sin(x[i-1] * math.pi / 2.0) for i in range(1, len(x)-m+1)],
                    1.0)
    
    if m != 1:
        result *= math.cos(x[len(x)-m] * math.pi / 2.0)
        
    return _correct_to_01(result)

def _linear(x, m):
    result = functools.reduce(operator.mul, x[:len(x)-m], 1.0)
    
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

def _transform(x, M, lam, nvars, nobjs):
    k = nvars - nobjs + 1
    p = [0.0]*nvars
    psum = [0.0]*nobjs
    zz = [0.0]*nvars
    
    for i in range(nvars):
        z = sum([M[i][j]*x[j] for j in range(nvars)])
        
        if z >= 0 and z <= 1:
            zz[i] = z
            p[i] = 0.0
        elif z < 0:
            zz[i] = -lam[i]*z
            p[i] = -z
        else:
            zz[i] = 1.0 - lam[i]*(z - 1.0)
            p[i] = z - 1.0
            
    for i in range(nvars-k+1, nvars+1):
        for j in range(nobjs):
            psum[j] = math.sqrt(math.pow(psum[j], 2.0) + math.pow(p[i-1], 2.0))
            
    for i in range(1, nobjs+1):
        for j in range(nobjs-i, 0, -1):
            psum[i-1] = math.sqrt(math.pow(psum[i-1], 2.0) + math.pow(p[j-1], 2.0))
            
        if i > 1:
            psum[i-1] = math.sqrt(math.pow(psum[i-1], 2.0) + math.pow(p[nobjs-i], 2.0))
            
    return zz, psum

class UF1(Problem):
    
    def __init__(self, nvars = 30):
        super(UF1, self).__init__(nvars, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-1, 1)
    
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

class UF6(Problem):
    
    def __init__(self, nvars = 30):
        super(UF6, self).__init__(nvars, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-1, 1)
    
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        sum1 = 0.0
        sum2 = 0.0
        prod1 = 1.0
        prod2 = 1.0
        N = 2.0
        E = 0.1
        
        for j in range(2, self.nvars+1):
            yj = x[j-1] - math.sin(6.0*math.pi*x[0] + j*math.pi/self.nvars)
            pj = math.cos(20.0*yj*math.pi/math.sqrt(j))
            
            if j % 2 == 1:
                sum1 += yj**2
                prod1 *= pj
                count1 += 1
            else:
                sum2 += yj**2
                prod2 *= pj
                count2 += 1
        
        hj = 2.0 * (0.5/N + E) * math.sin(2.0*N*math.pi*x[0])
        hj = max(hj, 0.0)
        
        f1 = x[0] + hj + 2.0*(4.0*sum1 - 2.0*prod1 + 2.0)/count1
        f2 = 1.0 - x[0] + hj + 2.0*(4.0*sum2 - 2.0*prod2 + 2.0)/count2
        solution.objectives[:] = [f1, f2]

class UF7(Problem):
    
    def __init__(self, nvars = 30):
        super(UF7, self).__init__(nvars, 2)
        self.types[0] = Real(0, 1)
        self.types[1:] = Real(-1, 1)
    
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
        
        yj = math.pow(x[0], 0.2)
        f1 = yj + 2.0*sum1/count1
        f2 = 1.0 - yj + 2.0*sum2/count2
        solution.objectives[:] = [f1, f2]

class UF8(Problem):
    
    def __init__(self, nvars = 30):
        super(UF8, self).__init__(nvars, 3)
        self.types[0:2] = Real(0, 1)
        self.types[2:] = Real(-2, 2)
    
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        count3 = 0
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        
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
        solution.objectives[:] = [f1, f2, f3]

class UF9(Problem):
    
    def __init__(self, nvars = 30):
        super(UF9, self).__init__(nvars, 3)
        self.types[0:2] = Real(0, 1)
        self.types[2:] = Real(-2, 2)
    
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        count3 = 0
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        E = 0.1
        
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
        
        yj = (1.0 + E) * (1.0 - 4.0*(2.0*x[0] - 1.0)**2)
        yj = max(yj, 0.0)
        f1 = 0.5*(yj + 2.0*x[0])*x[1] + 2.0*sum1/count1
        f2 = 0.5*(yj - 2.0*x[0] + 2.0)*x[1] + 2.0*sum2/count2
        f3 = 1.0 - x[1] + 2.0*sum3/count3
        solution.objectives[:] = [f1, f2, f3]

class UF10(Problem):
    
    def __init__(self, nvars = 30):
        super(UF10, self).__init__(nvars, 3)
        self.types[0:2] = Real(0, 1)
        self.types[2:] = Real(-2, 2)
    
    def evaluate(self, solution):
        x = solution.variables[:]
        count1 = 0
        count2 = 0
        count3 = 0
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        
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
        
        f1 = math.cos(0.5*math.pi*x[0])*math.cos(0.5*math.pi*x[1]) + 2.0*sum1/count1
        f2 = math.cos(0.5*math.pi*x[0])*math.sin(0.5*math.pi*x[1]) + 2.0*sum2/count2
        f3 = math.sin(0.5*math.pi*x[0]) + 2.0*sum3/count3
        solution.objectives[:] = [f1, f2, f3]
        
class UF11(Problem):
    
    LB = [ -1.773, -1.846, -1.053, -2.370, -1.603, -1.878, -1.677, -0.935,
                    -1.891, -0.964, -0.885, -1.690, -2.235, -1.541, -0.720,
                    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ]
    
    UB = [ 1.403, 1.562, 2.009, 0.976, 1.490, 1.334, 1.074, 2.354, 1.462,
                    2.372, 2.267, 1.309, 0.842, 1.665, 2.476, 1.000, 1.000,
                    1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                    1.000, 1.000, 1.000, 1.000, 1.000 ]
    
    M = [   [ 0.0128, 0.2165, 0.4374, -0.0800, 0.0886, -0.2015, 0.1071,
                    0.2886, 0.2354, 0.2785, -0.1748, 0.2147, 0.1649,
                    -0.3043, 0.5316, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.4813, 0.2420, -0.3663, -0.0420, -0.0088, -0.4945, -0.3073,
                    0.1990, 0.0441, -0.0627, 0.0191, 0.3880, -0.0618,
                    -0.0319, -0.1833, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0.000, 0.000 ],
            [ 0.4816, -0.2254, 0.0663, 0.4801, 0.2009, -0.0008, -0.1501,
                    0.0269, -0.2037, 0.4334, -0.2157, -0.3175, -0.0923,
                    0.1451, 0.1118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ -0.0876, -0.2667, -0.0063, 0.2114, 0.4506, 0.0823, -0.0125,
                    0.2313, 0.0840, -0.2376, 0.1938, -0.0030, 0.3391,
                    0.0863, 0.1231, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ -0.1025, 0.4011, -0.0117, 0.2076, 0.2585, 0.1124, -0.0288,
                    0.3095, -0.6146, -0.2376, 0.1938, -0.0030, 0.3391,
                    0.0863, 0.1231, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.4543, -0.2761, -0.2985, -0.2837, 0.0634, 0.1070, 0.2996,
                    -0.2690, -0.1634, -0.1452, 0.1799, -0.0014, 0.2394,
                    -0.2745, 0.3969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ -0.1422, -0.4364, 0.0751, -0.2235, 0.3966, -0.0252, 0.0908,
                    0.0477, -0.2254, 0.1801, -0.0552, 0.5770, -0.0396,
                    0.3765, -0.0522, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.3542, -0.2245, 0.3497, -0.1609, -0.1107, 0.0079, 0.2241,
                    0.4517, 0.1309, -0.3355, -0.1123, -0.1831, 0.3000,
                    0.2045, -0.3191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.0005, 0.0377, -0.2808, -0.0641, 0.1316, 0.2191, 0.0207,
                    0.3308, 0.4117, 0.3839, 0.5775, -0.1219, 0.1192,
                    0.2435, 0.0414, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ -0.1177, -0.0001, -0.1992, -0.4533, 0.4234, -0.0191, -0.3740,
                    0.1325, 0.0972, -0.2042, -0.3493, -0.4018, -0.1087,
                    0.0918, 0.2217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.1818, 0.3022, -0.1388, -0.2380, -0.0773, 0.6463, 0.0450,
                    0.1030, -0.0958, 0.2837, -0.3969, 0.1779, -0.0251,
                    -0.1543, -0.2452, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0.000, 0.000 ],
            [ -0.1889, -0.4397, -0.2206, 0.0981, -0.5203, 0.1325, -0.3427,
                    0.4242, -0.1271, -0.0291, -0.0795, 0.1213, 0.0565,
                    -0.1092, 0.2720, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0 ],
            [ -0.1808, -0.0624, -0.2689, 0.2289, 0.1128, -0.0844, -0.0549,
                    -0.2202, 0.2450, 0.0825, -0.3319, 0.0513, 0.7523,
                    0.0043, -0.1472, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0 ],
            [ -0.0983, 0.0611, -0.4145, 0.3017, 0.0410, -0.0703, 0.6250,
                    0.2449, 0.1307, -0.1714, -0.3045, 0.0218, -0.2837,
                    0.1408, 0.1633, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0 ],
            [ 0.2026, 0.0324, 0.1496, 0.3129, 0.1437, 0.4331, -0.2629,
                    -0.1498, 0.3746, -0.4366, 0.0163, 0.3316, -0.0697,
                    0.1833, 0.2412, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ] ]
    
    LAM = [ 0.113, 0.105, 0.117, 0.119, 0.108, 0.110, 0.101, 0.107, 0.111,
            0.109, 0.120, 0.108, 0.101, 0.105, 0.116, 1.000, 1.000, 1.000,
            1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
            1.000, 1.000, 1.000 ]
    
    def __init__(self):
        super(UF11, self).__init__(30, 5)
        self.types[:] = [Real(UF11.LB[i], UF11.UB[i]) for i in range(self.nvars)]
        self.internal_problem = DTLZ2(self.nobjs, self.nvars)
        
    def evaluate(self, solution):
        zz, psum = _transform(solution.variables[:], UF11.M, UF11.LAM, self.nvars, self.nobjs)
        
        transformed_solution = Solution(self.internal_problem)
        transformed_solution.variables[:] = zz
        transformed_solution.evaluate()
        
        for i in range(self.nobjs):
            solution.objectives[i] = [2.0 / (1.0 + math.exp(-psum[i])) * (transformed_solution.objectives[i] + 1.0)]

class UF12(Problem):
    
    LB = [ -1.773, -1.846, -1.053, -2.370, -1.603, -1.878, -1.677, -0.935,
                    -1.891, -0.964, -0.885, -1.690, -2.235, -1.541, -0.720,
                    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ]
    
    UB = [ 1.403, 1.562, 2.009, 0.976, 1.490, 1.334, 1.074, 2.354, 1.462,
                    2.372, 2.267, 1.309, 0.842, 1.665, 2.476, 1.000, 1.000,
                    1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                    1.000, 1.000, 1.000, 1.000, 1.000 ]
    
    M = [   [ -0.1565, -0.2418, 0.5427, -0.2191, 0.2522, -0.0563, 0.1991,
                    0.1166, 0.2140, -0.0973, -0.0755, 0.4073, 0.4279,
                    -0.1876, -0.0968, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0.000, 0.000 ],
            [ 0.1477, -0.2396, -0.0022, 0.4180, 0.2675, -0.1365, -0.0729,
                    0.4761, -0.0685, 0.2105, 0.1388, 0.1465, -0.0256,
                    0.0292, 0.5767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.0322, 0.3727, -0.0467, 0.1651, -0.0672, 0.0638, -0.1168,
                    0.4055, 0.6714, -0.1948, -0.1451, 0.1734, -0.2788,
                    -0.0769, -0.1433, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0.000, 0.000 ],
            [ -0.3688, 0.1935, 0.3691, 0.4298, 0.2340, 0.2593, -0.3081,
                    -0.2013, -0.2779, -0.0932, 0.0003, 0.0149, -0.2303,
                    -0.3261, -0.0517, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0.000, 0.000 ],
            [ 0.0580, -0.0609, 0.0004, -0.1831, 0.0003, 0.4742, -0.2530,
                    -0.0750, 0.0839, 0.1606, 0.6020, 0.4103, -0.0857,
                    0.2954, -0.0819, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ -0.2145, -0.0056, -0.0251, 0.2288, -0.4870, -0.5486, 0.1253,
                    -0.1512, -0.0390, 0.0722, 0.3074, 0.4160, -0.1304,
                    -0.1610, -0.0848, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0.000, 0.000 ],
            [ 0.2557, -0.1087, 0.0679, -0.3120, 0.3567, -0.4644, -0.3535,
                    0.1060, -0.2158, -0.1330, -0.0154, 0.0911, -0.4154,
                    0.0356, -0.3085, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.2303, 0.4996, 0.1883, 0.1870, 0.1850, -0.0216, 0.4409,
                    -0.0573, -0.2396, 0.1471, -0.1540, 0.2731, -0.0398,
                    0.4505, -0.1131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ -0.1576, -0.0023, 0.2588, 0.2105, 0.2250, -0.2978, 0.0175,
                    -0.1157, 0.3717, 0.0562, 0.4068, -0.5081, 0.0718,
                    0.3443, -0.1488, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.1047, -0.0568, -0.2771, 0.3803, 0.0046, 0.0188, -0.1500,
                    0.2053, -0.2290, -0.4582, 0.1191, 0.0639, 0.4946,
                    0.1121, -0.4018, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0.000, 0.000 ],
            [ 0.3943, -0.0374, 0.3004, 0.1472, -0.2988, 0.0443, -0.2483,
                    0.1350, -0.0160, 0.5834, -0.1095, -0.1398, 0.1711,
                    -0.1867, -0.3518, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0.000, 0.000 ],
            [ 0.1244, -0.6134, 0.1823, 0.3012, -0.1968, 0.1616, 0.1025,
                    -0.1972, 0.1162, -0.2079, -0.3062, 0.0585, -0.3286,
                    0.3187, -0.0812, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0 ],
            [ 0.1832, -0.1559, -0.4327, 0.2059, 0.4677, 0.0317, 0.2233,
                    -0.3589, 0.2393, 0.2468, 0.0148, 0.1193, -0.0279,
                    -0.3600, -0.2261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0 ],
            [ 0.5027, 0.1935, 0.1571, 0.0503, -0.0503, -0.1443, -0.3080,
                    -0.4939, 0.1847, -0.2762, 0.0042, 0.0960, 0.2239,
                    -0.0579, 0.3840, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0 ],
            [ 0.3948, -0.0002, 0.2172, -0.0293, -0.0835, 0.1614, 0.4559,
                    0.1626, -0.1155, -0.3087, 0.4331, -0.2223, -0.2213,
                    -0.3658, -0.0188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ] ]
    
    LAM = [ 0.113, 0.105, 0.117, 0.119, 0.108, 0.110, 0.101, 0.107, 0.111,
            0.109, 0.120, 0.108, 0.101, 0.105, 0.116, 1.000, 1.000, 1.000,
            1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
            1.000, 1.000, 1.000 ]
    
    def __init__(self):
        super(UF12, self).__init__(30, 5)
        self.types[:] = [Real(UF11.LB[i], UF11.UB[i]) for i in range(self.nvars)]
        self.internal_problem = DTLZ3(self.nobjs, self.nvars)
        
    def evaluate(self, solution):
        zz, psum = _transform(solution.variables[:], UF11.M, UF11.LAM, self.nvars, self.nobjs)
        
        transformed_solution = Solution(self.internal_problem)
        transformed_solution.variables[:] = zz
        transformed_solution.evaluate()
        
        for i in range(self.nobjs):
            solution.objectives[i] = [2.0 / (1.0 + math.exp(-psum[i])) * (transformed_solution.objectives[i] + 1.0)]

class UF13(WFG):
    
    def __init__(self):
        super(UF13, self).__init__(8, 22, 5)
       
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

class CF1(Problem):
    
    def __init__(self, nvars = 10):
        super(CF1, self).__init__(nvars, 2, 1)
        self.types[:] = Real(0, 1)
        self.constraints[:] = ">=0"
     
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
        self.types[0:2] = Real(0, 1)
        self.types[2:] = Real(-4, 4)
        self.constraints[:] = ">=0"
        
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
        self.types[0:2] = Real(0, 1)
        self.types[2:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
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
        self.types[0:2] = Real(0, 1)
        self.types[2:] = Real(-2, 2)
        self.constraints[:] = ">=0"
        
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
          
################################################################################
# ZDT Problems
################################################################################

class ZDT(Problem):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, nvars):
        super(ZDT, self).__init__(nvars, 2)
        self.types[:] = Real(0, 1)
        
class ZDT1(ZDT):
    
    def __init__(self):
        super(ZDT1, self).__init__(30)
        
    def evaluate(self, solution):
        x = solution.variables[:]
        g = (9.0 / (self.nvars - 1.0))*sum(x[1:]) + 1.0
        h = 1.0 - math.sqrt(x[0] / g)
        solution.objectives[:] = [x[0], g*h]

class ZDT2(ZDT):
    
    def __init__(self):
        super(ZDT2, self).__init__(30)
        
    def evaluate(self, solution):
        x = solution.variables[:]
        g = (9.0 / (self.nvars - 1.0))*sum(x[1:]) + 1.0
        h = 1.0 - math.pow(x[0] / g, 2.0)
        solution.objectives[:] = [x[0], g*h]

class ZDT3(ZDT):
    
    def __init__(self):
        super(ZDT3, self).__init__(30)
        
    def evaluate(self, solution):
        x = solution.variables[:]
        g = (9.0 / (self.nvars - 1.0))*sum(x[1:]) + 1.0
        h = 1.0 - math.sqrt(x[0]/g) - (x[0]/g)*math.sin(10.0*math.pi*x[0])
        solution.objectives[:] = [x[0], g*h]
                              
class ZDT4(ZDT):
    
    def __init__(self):
        super(ZDT4, self).__init__(10)
        
    def evaluate(self, solution):
        x = solution.variables[:]
        g = 1.0 + 10.0*(self.nvars-1) + sum([math.pow(x[i], 2.0) - 10.0*math.cos(4.0*math.pi*x[i]) for i in range(1, self.nvars)])
        h = 1.0 - math.sqrt(x[0] / g)
        solution.objectives[:] = [x[0], g*h]
        
class ZDT5(ZDT):
    
    def __init__(self):
        super(ZDT5, self).__init__(11)
        self.types[0] = Binary(30)
        self.types[1:] = Binary(5)
        
    def evaluate(self, solution):
        f = 1.0 + sum(solution.variables[0])
        g = sum([2+sum(v) if sum(v) < 5 else 1 for v in solution.variables[1:]])
        h = 1.0 / f
        solution.objectives[:] = [f, g*h]
        
class ZDT6(ZDT):
    
    def __init__(self):
        super(ZDT6, self).__init__(10)
        
    def evaluate(self, solution):
        x = solution.variables[:]
        f = 1.0 - math.exp(-4.0*x[0])*math.pow(math.sin(6.0*math.pi*x[0]), 6.0)
        g = 1.0 + 9.0*math.pow(sum(x[1:]) / (self.nvars-1.0), 0.25)
        h = 1.0 - math.pow(x[0] / g, 2.0)
        solution.objectives[:] = [f, g*h]
        