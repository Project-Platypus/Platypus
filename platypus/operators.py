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

import copy
import math
import random
from typing import List

import numpy as np

from .core import PlatypusError, Solution, ParetoDominance, Generator, Selector, Variator, \
                  Mutation, EPSILON, RankDominance
from .types import Real, Binary, Permutation, Subset
from .tools import add, subtract, multiply, is_zero, magnitude, orthogonalize, normalize, random_vector, zeros, roulette


def clip(value, min_value, max_value):
    return max(min_value, min(value, max_value))

class RandomGenerator(Generator):

    def __init__(self):
        super(RandomGenerator, self).__init__()

    def generate(self, problem):
        solution = Solution(problem)
        solution.variables = [x.rand() for x in problem.types]
        return solution

class InjectedPopulation(Generator):

    def __init__(self, solutions):
        super(InjectedPopulation, self).__init__()
        self.solutions = []

        for solution in solutions:
            self.solutions.append(copy.deepcopy(solution))

    def generate(self, problem):
        if len(self.solutions) > 0:
            # If we have more solutions to inject, return one from the list
            return self.solutions.pop()
        else:
            # Otherwise generate a random solution
            solution = Solution(problem)
            solution.variables = [x.rand() for x in problem.types]
            return solution

class TournamentSelector(Selector):

    def __init__(self, tournament_size = 2, dominance=ParetoDominance()):
        super(TournamentSelector, self).__init__(dominance)
        self.tournament_size = tournament_size
        self.dominance = dominance

    def select_one(self, population):
        winner = random.choice(population)

        for _ in range(self.tournament_size-1):
            candidate = random.choice(population)
            flag = self.dominance.compare(winner, candidate)

            if flag > 0:
                winner = candidate

        return winner

class RankSelector(Selector):

    def __init__(self, tournament_size = 2, dominance = RankDominance()):
        super(RankSelector, self).__init__()
        self.tournament_size = tournament_size
        self.dominance = dominance

    def select_one(self, population):
        winner = random.choice(population)

        for _ in range(self.tournament_size-1):
            candidate = random.choice(population)
            flag = self.dominance.compare(winner, candidate)

            if flag > 0:
                winner = candidate

        return winner

class PM(Mutation):

    def __init__(self, probability = 1, distribution_index = 20.0):
        super(PM, self).__init__()
        self.probability = probability
        self.distribution_index = distribution_index

    def mutate(self, parent):
        child = copy.deepcopy(parent)
        problem = child.problem
        probability = self.probability

        if isinstance(probability, int):
            probability /= float(len([t for t in problem.types if isinstance(t, Real)]))

        for i in range(len(child.variables)):
            if isinstance(problem.types[i], Real):
                if random.uniform(0.0, 1.0) <= probability:
                    child.variables[i] = self.pm_mutation(float(child.variables[i]),
                                                          problem.types[i].min_value,
                                                          problem.types[i].max_value)

                    child.evaluated = False

        return child

    def pm_mutation(self, x, lb, ub):
        u = random.uniform(0, 1)
        dx = ub - lb

        if u < 0.5:
            bl = (x - lb) / dx
            b = 2.0*u + (1.0 - 2.0*u)*pow(1.0 - bl, self.distribution_index + 1.0)
            delta = pow(b, 1.0 / (self.distribution_index + 1.0)) - 1.0
        else:
            bu = (ub - x) / dx
            b = 2.0*(1.0 - u) + 2.0*(u - 0.5)*pow(1.0 - bu, self.distribution_index + 1.0)
            delta = 1.0 - pow(b, 1.0 / (self.distribution_index + 1.0))

        x = x + delta*dx
        x = clip(x, lb, ub)

        return x

class SBX(Variator):

    def __init__(self, probability = 1.0, distribution_index = 15.0):
        super(SBX, self).__init__(2)
        self.probability = probability
        self.distribution_index = distribution_index

    def evolve(self, parents):
        child1 = copy.deepcopy(parents[0])
        child2 = copy.deepcopy(parents[1])

        if random.uniform(0.0, 1.0) <= self.probability:
            problem = child1.problem
            nvars = problem.nvars

            for i in range(nvars):
                if isinstance(problem.types[i], Real):
                    if random.uniform(0.0, 1.0) <= 0.5:
                        x1 = float(child1.variables[i])
                        x2 = float(child2.variables[i])
                        lb = problem.types[i].min_value
                        ub = problem.types[i].max_value

                        x1, x2 = self.sbx_crossover(x1, x2, lb, ub)

                        child1.variables[i] = x1
                        child2.variables[i] = x2
                        child1.evaluated = False
                        child2.evaluated = False

        return [child1, child2]

    def sbx_crossover(self, x1, x2, lb, ub):
        dx = x2 - x1

        if dx > EPSILON:
            if x2 > x1:
                y2 = x2
                y1 = x1
            else:
                y2 = x1
                y1 = x2

            beta = 1.0 / (1.0 + (2.0 * (y1 - lb) / (y2 - y1)))
            alpha = 2.0 - pow(beta, self.distribution_index + 1.0)
            rand = random.uniform(0.0, 1.0)

            if rand <= 1.0 / alpha:
                alpha = alpha * rand
                betaq = pow(alpha, 1.0 / (self.distribution_index + 1.0))
            else:
                alpha = alpha * rand;
                alpha = 1.0 / (2.0 - alpha)
                betaq = pow(alpha, 1.0 / (self.distribution_index + 1.0))

            x1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            beta = 1.0 / (1.0 + (2.0 * (ub - y2) / (y2 - y1)));
            alpha = 2.0 - pow(beta, self.distribution_index + 1.0);

            if rand <= 1.0 / alpha:
                alpha = alpha * rand;
                betaq = pow(alpha, 1.0 / (self.distribution_index + 1.0));
            else:
                alpha = alpha * rand;
                alpha = 1.0 / (2.0 - alpha);
                betaq = pow(alpha, 1.0 / (self.distribution_index + 1.0));

            x2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

            # randomly swap the values
            if bool(random.getrandbits(1)):
                x1, x2 = x2, x1

            x1 = clip(x1, lb, ub)
            x2 = clip(x2, lb, ub)

        return x1, x2

class GAOperator(Variator):

    def __init__(self, variation, mutation):
        super(GAOperator, self).__init__(variation.arity)
        self.variation = variation
        self.mutation = mutation
        self.i = 0

    def evolve(self, parents):
        if self.i == 0:
            self.i +=1
        children = self.variation.evolve(parents)
        children_mutated = list(map(self.mutation.evolve, children))
        a=1
        return children_mutated


class CompoundMutation(Mutation):

    def __init__(self, *mutators):
        super(CompoundMutation, self).__init__()
        self.mutators = mutators

    def mutate(self, parent):
        result = parent

        for mutator in self.mutators:
            result = mutator.mutate(result)

        return result

class CompoundOperator(Variator):

    def __init__(self, *variators):
        super(CompoundOperator, self).__init__(variators[0].arity)
        self.variators = variators

    def evolve(self, parents):
        offspring = parents

        for variator in self.variators:
            if variator.arity == len(offspring):
                offspring = variator.evolve(offspring)
            elif variator.arity == 1 and len(offspring) >= 1:
                offspring = list(map(variator.evolve, offspring))
            else:
                raise PlatypusError("unexpected number of offspring, expected %d, received %d" % (variator.arity, len(offspring)))

        return offspring

class DifferentialEvolution(Variator):

    def __init__(self, crossover_rate=0.1, step_size=0.5):
        super(DifferentialEvolution, self).__init__(4)
        self.crossover_rate = crossover_rate
        self.step_size = step_size

    def evolve(self, parents):
        result = copy.deepcopy(parents[0])
        problem = result.problem
        jrand = random.randrange(problem.nvars)

        for j in range(problem.nvars):
            if random.uniform(0.0, 1.0) <= self.crossover_rate or j == jrand:
                v1 = float(parents[1].variables[j])
                v2 = float(parents[2].variables[j])
                v3 = float(parents[3].variables[j])

                y = v3 + self.step_size*(v1 - v2)
                y = clip(y, problem.types[j].min_value, problem.types[j].max_value)

                result.variables[j] = y
                result.evaluated = False

        return [result]

class UniformMutation(Mutation):

    def __init__(self, probability: float, perturbation: float, vertical: bool):

        super(UniformMutation, self).__init__()
        self.probability = probability
        self.perturbation = perturbation
        self.vertical = vertical
        self.n_points = int(PoddieParameters.get_instance().route['n_points'])

    def mutate(self, parent):
        result = parent.from_instance()
        problem = result.problem

        for i in range(self.n_points):
            i = i+self.n_points if self.vertical else i
            if random.uniform(0.0, 1.0) < self.probability:
                type = problem.types[i]
                value = result.variables[i] + (random.uniform(0.0, 1.0) - 0.5) * self.perturbation
                result.variables[i] = clip(value, type.min_value, type.max_value)
                result.evaluated = False

        return result

class SingleUniformMutation(Mutation):

    def __init__(self, probability:float, perturbation: float, vertical:bool):
        super(SingleUniformMutation, self).__init__()
        self.probability = probability
        self.perturbation = perturbation
        self.vertical = vertical
        self.n_points = int(PoddieParameters.get_instance().route['n_points'])

    def mutate(self, parent):
        result = parent.from_instance()
        problem = result.problem

        i = random.randint(0, self.n_points-1)
        i = i+self.n_points if self.vertical else i
        if random.uniform(0.0, 1.0) < self.probability:
            type = problem.types[i]
            value = result.variables[i] + (random.uniform(0.0, 1.0) - 0.5) * self.perturbation
            result.variables[i] = clip(value, type.min_value, type.max_value)
            result.evaluated = False

        return result

class NonUniformMutation(Mutation):

    def __init__(self, probability, perturbation, max_iterations, vertical=False):
        super(NonUniformMutation, self).__init__()
        self.probability = probability
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.vertical = vertical
        self.n_points = int(PoddieParameters.get_instance().route['n_points'])
        self.ngen_max = PoddieParameters.get_instance().optimisation['generation_size']

    def _delta(self, difference):
        current_iteration = self.ngen / self.ngen_max
        fraction = min(1.0, current_iteration / float(self.max_iterations))
        return difference * (1.0 - math.pow(random.uniform(0.0, 1.0), math.pow(1.0 - fraction,
                                                                               self.perturbation)))

    def mutate(self, parent):
        result = parent.from_instance()
        problem = result.problem

        for i in range(self.n_points):
            i = i + self.n_points if self.vertical else i
            if random.uniform(0.0, 1.0) <= self.probability:
                type = problem.types[i]
                value = result.variables[i]

                if bool(random.getrandbits(1)):
                    value += self._delta(type.max_value - value)
                else:
                    value += self._delta(type.min_value - value)

                # result.variables[i] = clip(value, type.min_value, type.max_value)
                result.evaluated = False

        return result


class UM(Mutation):
    """Uniform mutation."""

    def __init__(self, probability = 1):
        super(UM, self).__init__()
        self.probability = probability

    def mutate(self, parent):
        child = copy.deepcopy(parent)
        problem = child.problem
        probability = self.probability

        if isinstance(probability, int):
            probability /= float(len([t for t in problem.types if isinstance(t, Real)]))

        for i in range(len(child.variables)):
            if isinstance(problem.types[i], Real):
                if random.uniform(0.0, 1.0) <= self.probability:
                    child.variables[i] = self.um_mutation(float(child.variables[i]),
                                                          problem.types[i].min_value,
                                                          problem.types[i].max_value)
                    child.evaluated = False

        return child

    def um_mutation(self, x, lb, ub):
        return random.uniform(lb, ub)

class PCX(Variator):

    def __init__(self, nparents = 10, noffspring = 2, eta = 0.1, zeta = 0.1):
        super(PCX, self).__init__(nparents)
        self.nparents = nparents
        self.noffspring = noffspring
        self.eta = eta
        self.zeta = zeta

    def evolve(self, parents):
        result = []

        for _ in range(self.noffspring):
            index = random.randrange(len(parents))
            parents[index], parents[-1] = parents[-1], parents[index]

            result.append(self.pcx(parents))

        return result

    def pcx(self, parents):
        k = len(parents)
        n = parents[0].problem.nvars
        x = []

        for i in range(k):
            x.append(parents[i].variables[:])

        g = [sum([x[i][j] for i in range(k)]) / k for j in range(n)]
        D = 0.0

        # basis vectors defined by parents
        e_eta = []
        e_eta.append(subtract(x[k-1], g))

        for i in range(k-1):
            d = subtract(x[i], g)

            if not is_zero(d):
                e = orthogonalize(d, e_eta)

                if not is_zero(e):
                    D += magnitude(e)
                    e_eta.append(normalize(e))

        D /= k-1

        # construct the offspring
        variables = x[k-1]
        variables = add(variables, multiply(random.gauss(0.0, self.zeta), e_eta[0]))

        eta = random.gauss(0.0, self.eta)

        for i in range(1, len(e_eta)):
            variables = add(variables, multiply(eta*D, e_eta[i]))

        result = copy.deepcopy(parents[k-1])

        for j in range(n):
            type = result.problem.types[j]
            result.variables[j] = clip(variables[j], type.min_value, type.max_value)

        result.evaluated = False
        return result

class UNDX(Variator):

    def __init__(self, nparents = 10, noffspring = 2, zeta = 0.5, eta = 0.35):
        super(UNDX, self).__init__(nparents)
        self.nparents = nparents
        self.noffspring = noffspring
        self.zeta = zeta
        self.eta = eta

    def evolve(self, parents):
        result = []

        for _ in range(self.noffspring):
            result.append(self.undx(parents))

        return result

    def undx(self, parents):
        k = len(parents)
        n = parents[0].problem.nvars
        x = []

        for i in range(k):
            x.append(parents[i].variables[:])

        g = [sum([x[i][j] for i in range(k)]) / k for j in range(n)]


        # basis vectors defined by parents
        e_zeta = []
        e_eta = []

        for i in range(k-1):
            d = subtract(x[i], g)

            if not is_zero(d):
                dbar = magnitude(d)
                e = orthogonalize(d, e_zeta)

                if not is_zero(e):
                    e_zeta.append(multiply(dbar, normalize(e)))

        D = magnitude(subtract(x[k-1], g))

        # create the remaining basis vectors
        for i in range(n-len(e_zeta)):
            d = random_vector(n)

            if not is_zero(d):
                e = orthogonalize(d, e_eta)

                if not is_zero(e):
                    e_eta.append(multiply(D, normalize(e)))

        # construct the offspring
        variables = g

        for i in range(len(e_zeta)):
            variables = add(variables, multiply(random.gauss(0.0, self.zeta), e_zeta[i]))

        for i in range(1, len(e_eta)):
            variables = add(variables, multiply(random.gauss(0.0, self.eta / math.sqrt(n)), e_eta[i]))

        result = copy.deepcopy(parents[k-1])

        for j in range(n):
            type = result.problem.types[j]
            result.variables[j] = clip(variables[j], type.min_value, type.max_value)

        result.evaluated = False
        return result

class SPX(Variator):

    def __init__(self, nparents = 10, noffspring = 2, expansion = None):
        super(SPX, self).__init__(nparents)
        self.nparents = nparents
        self.noffspring = noffspring

        if expansion is None:
            self.expansion = math.sqrt(nparents+1)
        else:
            self.expansion = expansion

    def evolve(self, parents):
        n = len(parents)
        m = parents[0].problem.nvars
        x = []

        for i in range(n):
            x.append(parents[i].variables[:])

        # compute center of mass
        G = [sum([x[i][j] for i in range(n)]) / n for j in range(m)]

        # compute expanded simplex vertices
        for i in range(n):
            x[i] = add(G, multiply(self.expansion, subtract(x[i], G)))

        # generate offspring
        result = []

        for _ in range(self.noffspring):
            child = copy.deepcopy(parents[n-1])
            r = [math.pow(random.uniform(0.0, 1.0), 1.0 / (i + 1.0)) for i in range(n-1)]
            C = zeros(n, m)

            for i in range(n):
                for j in range(m):
                    if i == 0:
                        C[i][j] = 0.0
                    else:
                        C[i][j] = r[i-1] * (x[i-1][j] - x[i][j] + C[i-1][j])

            for j in range(m):
                type = child.problem.types[j]
                child.variables[j] = clip(x[n-1][j] + C[n-1][j], type.min_value, type.max_value)

            child.evaluated = False
            result.append(child)

        return result

class BitFlip(Mutation):

    def __init__(self, probability=1):
        """Bit Flip Mutation for Binary Strings.

        Parameters
        ----------
        probability : int or float
            The probability of flipping an individual bit.  If the value is
            an int, then the probability is divided by the number of bits.
        """
        super(BitFlip, self).__init__()
        self.probability = probability

    def mutate(self, parent):
        result = copy.deepcopy(parent)
        problem = result.problem
        probability = self.probability

        if isinstance(probability, int):
            probability /= sum([t.nbits for t in problem.types if isinstance(t, Binary)])

        for i in range(problem.nvars):
            type = problem.types[i]

            if isinstance(type, Binary):
                for j in range(type.nbits):
                    if random.uniform(0.0, 1.0) <= probability:
                        result.variables[i][j] = not result.variables[i][j]
                        result.evaluated = False

        return result

class HUX(Variator):

    def __init__(self, probability = 1.0):
        super(HUX, self).__init__(2)
        self.probability = probability

    def evolve(self, parents):
        result1 = copy.deepcopy(parents[0])
        result2 = copy.deepcopy(parents[1])
        problem = result1.problem

        if random.uniform(0.0, 1.0) <= self.probability:
            for i in range(problem.nvars):
                if isinstance(problem.types[i], Binary):
                    for j in range(problem.types[i].nbits):
                        if result1.variables[i][j] != result2.variables[i][j]:
                            if bool(random.getrandbits(1)):
                                result1.variables[i][j] = not result1.variables[i][j]
                                result2.variables[i][j] = not result2.variables[i][j]
                                result1.evaluated = False
                                result2.evaluated = False

        return [result1, result2]

class Swap(Mutation):

    def __init__(self, probability = 0.3):
        super(Swap, self).__init__()
        self.probability = probability

    def mutate(self, parent):
        result = copy.deepcopy(parent)
        problem = result.problem

        for index in range(problem.nvars):
            if isinstance(problem.types[index], Permutation) and random.uniform(0.0, 1.0) <= self.probability:
                permutation = result.variables[index]
                i = random.randrange(len(permutation))
                j = random.randrange(len(permutation))

                if len(permutation) > 1:
                    while i == j:
                        j = random.randrange(len(permutation))

                permutation[i], permutation[j] = permutation[j], permutation[i]
                result.evaluated = False

        return result

class PMX(Variator):

    def __init__(self, probability = 1.0):
        super(PMX, self).__init__(2)
        self.probability = probability

    def evolve(self, parents):
        result1 = copy.deepcopy(parents[0])
        result2 = copy.deepcopy(parents[1])
        problem = result1.problem

        for index in range(problem.nvars):
            if isinstance(problem.types[index], Permutation) and random.uniform(0.0, 1.0) <= self.probability:
                p1 = result1.variables[index]
                p2 = result2.variables[index]
                n = len(p1)
                o1 = [None]*n
                o2 = [None]*n

                # select cutting points
                cp1 = random.randrange(n)
                cp2 = random.randrange(n)

                if n > 1:
                    while cp1 == cp2:
                        cp2 = random.randrange(n)

                if cp1 > cp2:
                    cp1, cp2 = cp2, cp1

                # exchange between the cutting points, setting up replacement arrays
                replacement1 = {}
                replacement2 = {}

                for i in range(cp1, cp2+1):
                    o1[i] = p2[i]
                    o2[i] = p1[i]
                    replacement1[p2[i]] = p1[i]
                    replacement2[p1[i]] = p2[i]

                # fill in remaining slots with replacements
                for i in range(n):
                    if i < cp1 or i > cp2:
                        n1 = p1[i]
                        n2 = p2[i]

                        while n1 in replacement1:
                            n1 = replacement1[n1]

                        while n2 in replacement2:
                            n2 = replacement2[n2]

                        o1[i] = n1
                        o2[i] = n2

                result1.variables[index] = o1
                result2.variables[index] = o2
                result1.evaluated = False
                result2.evaluated = False

        return [result1, result2]

class Insertion(Mutation):

    def __init__(self, probability = 0.3):
        super(Insertion, self).__init__()
        self.probability = probability

    def mutate(self, parent):
        result = copy.deepcopy(parent)
        problem = result.problem

        for index in range(problem.nvars):
            if isinstance(problem.types[index], Permutation) and random.uniform(0.0, 1.0) <= self.probability:
                permutation = result.variables[index]
                i = random.randrange(len(permutation))
                j = random.randrange(len(permutation))

                if len(permutation) > 1:
                    while i == j:
                        j = random.randrange(len(permutation))

                # remove the i-th element and insert at j-th position
                temp = permutation[i]

                if i < j:
                    for k in range(i+1, j+1):
                        permutation[k-1] = permutation[k]
                elif i > j:
                    for k in range(i-1, j-1, -1):
                        permutation[k+1] = permutation[k]

                permutation[j] = temp
                result.evaluated = False

        return result

class Replace(Mutation):

    def __init__(self, probability = 0.3):
        super(Replace, self).__init__()
        self.probability = probability

    def mutate(self, parent):
        result = copy.deepcopy(parent)
        problem = result.problem

        for index in range(problem.nvars):
            if isinstance(problem.types[index], Subset) and random.uniform(0.0, 1.0) <= self.probability:
                subset = result.variables[index]

                if len(subset) < len(problem.types[index].elements):
                    i = random.randrange(len(subset))

                    nonmembers = list(set(problem.types[index].elements) - set(subset))
                    j = random.randrange(len(nonmembers))
                    subset[i] = nonmembers[j]
                    result.evaluated = False

        return result

class SSX(Variator):

    def __init__(self, probability = 1.0):
        super(SSX, self).__init__(2)
        self.probability = probability

    def evolve(self, parents):
        result1 = copy.deepcopy(parents[0])
        result2 = copy.deepcopy(parents[1])
        problem = result1.problem

        for i in range(problem.nvars):
            if isinstance(problem.types[i], Subset) and random.uniform(0.0, 1.0) <= self.probability:
                s1 = set(result1.variables[i])
                s2 = set(result2.variables[i])

                for j in range(problem.types[i].size):
                    if result2.variables[i][j] not in s1 and result1.variables[i][j] not in s2 and random.uniform(0.0, 1.0) < 0.5:
                        temp = result1.variables[i][j]
                        result1.variables[i][j] = result2.variables[i][j]
                        result2.variables[i][j] = temp

                result1.evaluated = False
                result2.evaluated = False

        return [result1, result2]

class Multimethod(Variator):

    def __init__(self, variators: List[Variator], tag: str, update_frequency=100):
        super(Multimethod, self).__init__(max([v.arity for v in variators]))
        self.variators = variators
        self.update_frequency = update_frequency
        self.last_update = 0
        self.probabilities = np.array([1.0 / len(variators) for _ in range(len(variators))])
        self.min_probability = 0.1 if len(variators) <= 10 else 1/len(variators)
        self._tag = tag

        self.select()

    def select(self):
        self.last_update += 1

        if self.last_update >= self.update_frequency:
            self.last_update = 0
            counts = [1 for _ in range(len(self.variators))]

            if self.population is not None:
                for solution in self.population:
                    if hasattr(solution, "operator") and self._tag in solution.operator.keys():
                        counts[solution.operator[self._tag]] += 1

            self.probabilities = np.array(
                               [counts[i] / float(sum(counts)) for i in range(len(self.variators))])

            probabilities_less_than_min = self.probabilities < self.min_probability
            if np.any(probabilities_less_than_min):
                print(f"modifying probabilities to minimum {self.min_probability}")
                self.probabilities[probabilities_less_than_min] = self.min_probability
                remaining_probability = 1-np.sum(self.probabilities[probabilities_less_than_min])
                self.probabilities[~probabilities_less_than_min] *= remaining_probability/\
                                            np.sum(self.probabilities[~probabilities_less_than_min])

            print(self.probabilities)

        self.next_variator = roulette(self.probabilities)
        self.arity = self.variators[self.next_variator].arity

    def evolve(self, parents):
        variator = self.variators[self.next_variator]
        result = variator.evolve(parents)

        if isinstance(result, list):
            for solution in result:
                if not hasattr(solution, "operator"):
                    solution.operator = dict()
                solution.operator[self._tag] = self.next_variator
        else:
            if not hasattr(result, "operator"):
                result.operator = dict()
            result.operator[self._tag] = self.next_variator

        self.select()
        return result

    def mutate(self, parents):
        """ Mutator interface"""
        return self.evolve(parents)
