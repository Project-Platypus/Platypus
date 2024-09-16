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
import json
import platypus
from argparse import ArgumentParser
from ._tools import only_keys_for, parse_cli_keyvalue, type_cast

parser = ArgumentParser(prog="platypus",
                        description="Platypus (platypus-opt) - Multobjective optimization in Python")

parser.add_argument("-v", "--version", action="version", version=platypus.__version__)

subparsers = parser.add_subparsers(title="commands", required=True, dest="command")

hypervolume_parser = subparsers.add_parser("hypervolume", help="compute hypervolume")
hypervolume_parser.add_argument("-r", "--reference", help="reference set")
hypervolume_parser.add_argument("--minimum", type=float, metavar="N", nargs="*", help="minimum bounds, optional")
hypervolume_parser.add_argument("--maximum", type=float, metavar="N", nargs="*", help="maximum bounds, optional")
hypervolume_parser.add_argument("filename")

gd_parser = subparsers.add_parser("gd", help="compute generaional distance")
gd_parser.add_argument("-r", "--reference", help="reference set", required=True)
gd_parser.add_argument("filename")

igd_parser = subparsers.add_parser("igd", help="compute inverted generaional distance")
igd_parser.add_argument("-r", "--reference", help="reference set", required=True)
igd_parser.add_argument("filename")

epsilon_parser = subparsers.add_parser("epsilon", help="compute additive epsilon indicator")
epsilon_parser.add_argument("-r", "--reference", help="reference set", required=True)
epsilon_parser.add_argument("filename")

spacing_parser = subparsers.add_parser("spacing", help="compute spacing")
spacing_parser.add_argument("filename")

solve_parser = subparsers.add_parser("solve", help="solve a built-in problem")
solve_parser.add_argument("-p", "--problem", help="name of the problem", required=True)
solve_parser.add_argument("-a", "--algorithm", help="name of the algorithm", required=True)
solve_parser.add_argument("-n", "--nfe", help="number of function evaluations", type=int, default=10000)
solve_parser.add_argument("-o", "--output", help="output filename")
solve_parser.add_argument("--problem_module", help="module containing the problem (if not built-in)")
solve_parser.add_argument("--algorithm_module", help="module containing the algorithm (if not built-in)")
solve_parser.add_argument("arguments", metavar="KEY=VALUE", nargs="*", help="additional arguments to set")

plot_parser = subparsers.add_parser("plot", help="generate simple 2D or 3D plot")
plot_parser.add_argument("-t", "--title", help="plot title")
plot_parser.add_argument("-o", "--output", help="output filename")
plot_parser.add_argument("filename")

args = parser.parse_args()

def load_set(file):
    try:
        return platypus.load_json(file)
    except json.decoder.JSONDecodeError:
        return platypus.load_objectives(file)

match args.command:
    case "hypervolume":
        ref_set = load_set(args.reference)
        input_set = load_set(args.filename)
        hyp = platypus.Hypervolume(reference_set=ref_set)
        print(hyp.calculate(input_set))
    case "gd":
        ref_set = load_set(args.reference)
        input_set = load_set(args.filename)
        gd = platypus.GenerationalDistance(reference_set=ref_set)
        print(gd.calculate(input_set))
    case "igd":
        ref_set = load_set(args.reference)
        input_set = load_set(args.filename)
        igd = platypus.InvertedGenerationalDistance(reference_set=ref_set)
        print(igd.calculate(input_set))
    case "epsilon":
        ref_set = load_set(args.reference)
        input_set = load_set(args.filename)
        eps = platypus.EpsilonIndicator(reference_set=ref_set)
        print(eps.calculate(input_set))
    case "spacing":
        input_set = load_set(args.filename)
        spacing = platypus.Spacing()
        print(spacing.calculate(input_set))
    case "solve":
        problem_module = __import__(args.problem_module if args.problem_module else "platypus", fromlist=[''])
        algorithm_module = __import__(args.algorithm_module if args.algorithm_module else "platypus", fromlist=[''])

        if args.problem not in dir(problem_module):
            raise platypus.PlatypusError(f"'{args.problem}' not found in module '{problem_module.__name__}'")
        if args.algorithm not in dir(algorithm_module):
            raise platypus.PlatypusError(f"'{args.algorithm}' not found in module '{algorithm_module.__name__}'")

        problem_class = getattr(problem_module, args.problem)
        algorithm_class = getattr(algorithm_module, args.algorithm)

        if not issubclass(problem_class, platypus.Problem):
            raise platypus.PlatypusError(f"'{args.problem}' is not a valid Problem")
        if not issubclass(algorithm_class, platypus.Algorithm):
            raise platypus.PlatypusError(f"'{args.algorithm}' is not a valid Algorithm")

        extra_args = parse_cli_keyvalue(args.arguments)
        problem = problem_class(**type_cast(only_keys_for(extra_args, problem_class), problem_class))
        algorithm = algorithm_class(problem, **type_cast(only_keys_for(extra_args, algorithm_class), algorithm_class))
        algorithm.run(args.nfe)

        if args.output:
            platypus.save_json(args.output, algorithm, indent=4)
        else:
            platypus.dump(algorithm.result, sys.stdout, indent=4)
    case "plot":
        import matplotlib.pyplot as plt
        input_set = load_set(args.filename)
        nobjs = input_set[0].problem.nobjs
        fig = plt.figure()

        if nobjs == 2:
            ax = fig.add_subplot()
            ax.scatter([s.objectives[0] for s in input_set],
                       [s.objectives[1] for s in input_set])
        elif nobjs == 3:
            ax = fig.add_subplot(projection='3d')
            ax.scatter([s.objectives[0] for s in input_set],
                       [s.objectives[1] for s in input_set],
                       [s.objectives[2] for s in input_set])
            ax.view_init(elev=30.0, azim=15.0)
        else:
            raise platypus.PlatypusError("plot requires a set with 2 or 3 objectives")

        ax.set_title(args.title if args.title else args.filename)
        ax.set_xlabel("$f_1(x)$")
        ax.set_ylabel("$f_2(x)$")

        if nobjs == 3:
            ax.set_zlabel("$f_3(x)$")

        if args.output:
            plt.savefig(args.output)
        else:
            plt.show()
