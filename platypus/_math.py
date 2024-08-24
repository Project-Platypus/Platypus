# Copyright 2015-2024 David Hadka
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

import sys
import math
import random
import operator
import functools
from functools import reduce
from .core import EPSILON
from .errors import SingularError


def clip(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def point_line_dist(point, line):
    return magnitude(subtract(multiply(float(dot(line, point))/float(dot(line, line)), line), point))

def magnitude(x):
    return math.sqrt(dot(x, x))

def add(x, y):
    return [x[i] + y[i] for i in range(len(x))]

def subtract(x, y):
    return [x[i] - y[i] for i in range(len(x))]

def multiply(s, x):
    return [s*x[i] for i in range(len(x))]

def dot(x, y):
    return reduce(operator.add, [x[i]*y[i] for i in range(len(x))], 0)

def is_zero(x):
    return all([abs(x[i]) < EPSILON for i in range(len(x))])

def project(u, v):
    return multiply(dot(u, v) / dot(v, v), v)

def orthogonalize(u, vs):
    for v in vs:
        u = subtract(u, project(u, v))
    return u

def normalize(u):
    if is_zero(u):
        raise ValueError("can not normalize a zero vector")
    return multiply(1.0 / magnitude(u), u)

def random_vector(n, rng=functools.partial(random.gauss, 0.0, 1.0)):
    return [rng() for _ in range(n)]

def zeros(m, n):
    return [[0.0]*n for _ in range(m)]

def lsolve(A, b):
    """Gaussian elimination with partial pivoting.

    This is implemented here to avoid a dependency on numpy.  This could be
    replaced by :code:`(x, _, _, _) = lstsq(A, b)`, but we prefer the pure
    Python implementation here.
    """
    N = len(b)

    for p in range(N):
        # find pivot row and swap
        max = p

        for i in range(p+1, N):
            if abs(A[i][p]) > abs(A[max][p]):
                max = i

        A[p], A[max] = A[max], A[p]
        b[p], b[max] = b[max], b[p]

        # singular or nearly singular
        if abs(A[p][p]) <= EPSILON:
            raise SingularError("matrix is singular or nearly singular")

        # pivot within A and b
        for i in range(p+1, N):
            alpha = A[i][p] / A[p][p]
            b[i] -= alpha * b[p]

            for j in range(p, N):
                A[i][j] -= alpha * A[p][j]

    # back substitution
    x = [0.0]*N

    for i in range(N-1, -1, -1):
        sum = 0.0

        for j in range(i+1, N):
            sum += A[i][j] * x[j]

        x[i] = (b[i] - sum) / A[i][i]

    return x

def choose(n, k):
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok / ktok
    else:
        return 0

def tred2(n, V, d, e):
    """Symmetric Householder reduction to tridiagonal form.

    This is derived from the Algol procedures tred2 by Bowdler, Martin,
    Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra,
    and the corresponding Fortran subroutine in EISPACK.
    """
    for j in range(n):
        d[j] = V[n-1][j]

    for i in range(n-1, 0, -1):
        scale = 0.0
        h = 0.0

        for k in range(i):
            scale += abs(d[k])

        if scale == 0.0:
            e[i] = d[i-1]

            for j in range(i):
                d[j] = V[i-1][j]
                V[i][j] = V[j][i] = 0.0

        else:
            for k in range(i):
                d[k] /= scale
                h += d[k]**2

            f = d[i-1]
            g = math.sqrt(h)

            if f > 0.0:
                g = -g

            e[i] = scale*g
            h -= f*g
            d[i-1] = f-g

            for j in range(i):
                e[j] = 0.0

            for j in range(i):
                f = d[j]
                V[j][i] = f
                g = e[j] + V[j][j]*f

                for k in range(j+1, i):
                    g += V[k][j]*d[k]
                    e[k] += V[k][j]*f

                e[j] = g

            f = 0.0

            for j in range(i):
                e[j] /= h
                f += e[j]*d[j]

            hh = f / (2*h)

            for j in range(i):
                e[j] -= hh*d[j]

            for j in range(i):
                f = d[j]
                g = e[j]

                for k in range(j, i):
                    V[k][j] -= f*e[k] + g*d[k]

                d[j] = V[i-1][j]
                V[i][j] = 0.0

        d[i] = h

    for i in range(n-1):
        V[n-1][i] = V[i][i]
        V[i][i] = 1.0
        h = d[i+1]

        if h != 0.0:
            for k in range(i+1):
                d[k] = V[k][i+1] / h

            for j in range(i+1):
                g = 0.0

                for k in range(i+1):
                    g += V[k][i+1] * V[k][j]

                for k in range(i+1):
                    V[k][j] -= g*d[k]

        for k in range(i+1):
            V[k][i+1] = 0.0

    for j in range(n):
        d[j] = V[n-1][j]
        V[n-1][j] = 0.0

    V[n-1][n-1] = 1.0
    e[0] = 0.0

def tql2(n, d, e, V):
    """Symmetric tridiagonal QL algorithm.

    This is derived from the Algol procedures tql2, by Bowdler, Martin,
    Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra,
    and the corresponding Fortran subroutine in EISPACK.
    """
    for i in range(1, n):
        e[i-1] = e[i]

    e[n-1] = 0.0

    f = 0.0
    tst1 = 0.0
    eps = math.pow(2.0, -52.0)

    for l in range(n):
        tst1 = max(tst1, abs(d[l]) + abs(e[l]))
        m = 1

        while m < n:
            if abs(e[m]) <= eps*tst1:
                break
            m += 1

        if m > l:
            iter = 0

            while True:
                iter += 1
                g = d[l]
                p = (d[l+1] - g) / (2.0*e[l])
                r = hypot(p, 1.0)

                if p < 0:
                    r = -r

                d[l] = e[l] / (p + r)
                d[l+1] = e[l] * (p + r)
                dl1 = d[l+1]
                h = g - d[l]

                for i in range(l+2, n):
                    d[i] -= h

                f += h
                p = d[m]
                c = 1.0
                c2 = c
                c3 = c
                el1 = e[l+1]
                s = 0.0
                s2 = 0.0

                for i in range(m-1, l-1, -1):
                    c3 = c2
                    c2 = c
                    s2 = s
                    g = c*e[i]
                    h = c*p
                    r = hypot(p, e[i])
                    e[i+1] = s*r
                    s = e[i] / r
                    c = p / r
                    p = c*d[i] - s*g
                    d[i+1] = h + s*(c*g + s*d[i])

                    for k in range(n):
                        h = V[k][i+1]
                        V[k][i+1] = s*V[k][i] + c*h
                        V[k][i] = c*V[k][i] - s*h

                p = -s*s2*c3*el1*e[l] / dl1
                e[l] = s*p
                d[l] = c*p

                if abs(e[l]) <= eps*tst1:
                    break

        d[l] = d[l] + f
        e[l] = 0.0

    for i in range(n-1):
        k = i
        p = d[i]

        for j in range(i+1, n):
            if d[j] < p:
                k = j
                p = d[j]

        if k != i:
            d[k] = d[i]
            d[i] = p

            for j in range(n):
                p = V[j][i]
                V[j][i] = V[j][k]
                V[j][k] = p

def hypot(a, b):
    """Computes sqrt(a**2 + b**2) without under/overflow."""
    if abs(a) > abs(b):
        r = b / a
        r = abs(a) * math.sqrt(1 + r*r)
    elif b != 0.0:
        r = a / b
        r = abs(b) * math.sqrt(1 + r*r)

    return r

def check_eigensystem(n, C, diag, Q):
    res = 0

    for i in range(n):
        for j in range(n):
            cc = 0.0
            dd = 0.0

            for k in range(n):
                cc += diag[k] * Q[i][k] * Q[j][k]
                dd += Q[i][k] * Q[j][k]

            if abs(cc - C[i if i > j else j][j if i > j else i])/math.sqrt(C[i][i]*C[j][j]) > 1e-10 and abs(cc - C[i if i > j else j][j if i > j else i]) > 1e-9:
                print("imprecise result detected", i, j, cc, C[i if i > j else j][j if i > j else i], (cc-C[i if i > j else j][j if i > j else i]), file=sys.stderr)
                res += 1

            if abs(dd - (1 if i == j else 0)) > 1e-10:
                print("imprecise result detected (Q not orthog.)", i, j, dd, file=sys.stderr)
                res += 1

    return res

def roulette(probabilities):
    """Performs roulette wheel selection given the probabilities.

    Given a list of probabilities, selects one of the items randomly.  The
    probabilities will be scaled if necessary, so the values do not need to
    sum to 1.0.  Returns the index of the selected item.

    Examples
    --------
        # Randomly selected between two items, preferring the first
        roulette([0.75, 0.25])

    Parameters
    ----------
    probabilities : list of float
        List of probabilities of selecting each item.
    """
    rand = random.uniform(0.0, sum(probabilities))
    value = 0.0

    for i in range(len(probabilities)):
        value += probabilities[i]

        if value > rand:
            return i

    return 0
