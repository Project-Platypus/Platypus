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
from platypus.core import Problem, Solution, EPSILON
from platypus.types import Real
from abc import ABCMeta

class DTLZ1(Problem):
    
    def __init__(self, nobjs = 2):
        super(DTLZ1, self).__init__(nobjs+4, nobjs)
        self.types[:] = Real(0, 1)
        
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
        solution.evaluated = True

class DTLZ2(Problem):
    
    def __init__(self, nobjs = 2):
        super(DTLZ2, self).__init__(nobjs+9, nobjs)
        self.types[:] = Real(0, 1)
        
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
        solution.evaluated = True
        
class DTLZ3(Problem):
    
    def __init__(self, nobjs = 2):
        super(DTLZ3, self).__init__(nobjs+9, nobjs)
        self.types[:] = Real(0, 1)
        
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
        solution.evaluated = True
        
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
            f[i] *= reduce(operator.mul,
                           [math.cos(0.5 * math.pi * math.pow(x, self.alpha)) for x in solution.variables[:self.nobjs-i-1]],
                           1)
            
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * math.pow(solution.variables[self.nobjs-i-1], self.alpha))
        
        solution.objectives[:] = f
        solution.evaluated = True
        
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
        solution.evaluated = True
        
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
        
    def evaluate(self, solution):
        y = _normalize_z(solution.variables[:])
        y = _WFG1_t1(y, self.k)
        y = _WFG1_t2(y, self.k)
        y = _WFG1_t3(y)
        y = _WFG1_t4(y, self.k, self.m)
        y = _WFG1_shape(y)
        
        solution.objectives[:] = y
        solution.evaluated = True
       
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
        solution.evaluated = True
  
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
        solution.evaluated = True
        
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
        solution.evaluated = True
        
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
        solution.evaluated = True
        
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
        solution.evaluated = True
        
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
        solution.evaluated = True
        
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
        solution.evaluated = True
        
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
        solution.evaluated = True
        
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