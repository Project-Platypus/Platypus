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
import importlib
import json
import locale
import logging
import os
import random
import re
import sys
from argparse import ArgumentParser

import platypus

from ._tools import (coalesce, log_args, only_keys_for, parse_cli_keyvalue,
                     type_cast)


def main(input=None):
    """The main entry point for the Platypus CLI."""
    LOGGER = logging.getLogger("Platypus")

    if input is None:
        input = sys.argv[1:]

    def load_set(file):
        """Loads input file from stdin or file."""
        if file is None:
            return platypus.load(sys.stdin)

        try:
            return platypus.load_json(file)
        except json.decoder.JSONDecodeError:
            return platypus.load_objectives(file)

    def save_set(result, file, indent=4):
        """Output result to stdout or file."""
        if file is None:
            platypus.dump(result, sys.stdout, indent=indent)
            sys.stdout.write(os.linesep)
        else:
            platypus.save_json(file, result, indent=indent)

    def split_list(type, separator=None):
        """Argparse type for comma-separated list of values.

        Accepts arguments of the form::

            --arg val1,val2,val3

        This is, in my opinion, a bit easier to use than argparse's default
        which will capture any subsequent positional arguments unless
        separated by ``--``.

        By default, supports either ``,`` or ``;`` as the separator, except
        in locales using that character as a decimal point.
        """
        separator = coalesce(separator, "".join({",", ";"}.difference(locale.localeconv()["decimal_point"])))
        pattern = re.compile(f"[{re.escape(separator)}]")
        return lambda input: [type(s.strip()) for s in pattern.split(input)]

    def debug_inputs(args):
        """Log CLI arguments."""
        for attr in dir(args):
            if not attr.startswith("_"):
                LOGGER.debug("Argument: %s=%s", attr, getattr(args, attr))

    parser = ArgumentParser(prog="platypus",
                            description="Platypus (platypus-opt) - Multobjective optimization in Python")

    parser.add_argument("-v", "--version", action="version", version=platypus.PlatypusConfig.version)
    parser.add_argument('--log', help='set the logging level', type=str.upper, default='WARNING',
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    subparsers = parser.add_subparsers(title="commands", required=True, dest="command")

    hypervolume_parser = subparsers.add_parser("hypervolume", help="compute hypervolume")
    hypervolume_parser.add_argument("-r", "--reference_set", help="reference set")
    hypervolume_parser.add_argument("--minimum", help="minimum bounds, optional", type=split_list(float))
    hypervolume_parser.add_argument("--maximum", help="maximum bounds, optional", type=split_list(float))
    hypervolume_parser.add_argument("filename", help="input filename", nargs="?")

    gd_parser = subparsers.add_parser("gd", help="compute generaional distance")
    gd_parser.add_argument("-r", "--reference_set", help="reference set", required=True)
    gd_parser.add_argument("filename", help="input filename", nargs="?")

    igd_parser = subparsers.add_parser("igd", help="compute inverted generaional distance")
    igd_parser.add_argument("-r", "--reference_set", help="reference set", required=True)
    igd_parser.add_argument("filename", help="input filename", nargs="?")

    epsilon_parser = subparsers.add_parser("epsilon", help="compute additive epsilon indicator")
    epsilon_parser.add_argument("-r", "--reference_set", help="reference set", required=True)
    epsilon_parser.add_argument("filename", help="input filename", nargs="?")

    spacing_parser = subparsers.add_parser("spacing", help="compute spacing")
    spacing_parser.add_argument("filename", help="input filename", nargs="?")

    solve_parser = subparsers.add_parser("solve", help="solve a built-in problem")
    solve_parser.add_argument("-p", "--problem", help="name of the problem", required=True)
    solve_parser.add_argument("-a", "--algorithm", help="name of the algorithm", required=True)
    solve_parser.add_argument("-n", "--nfe", help="number of function evaluations", type=int, default=10000)
    solve_parser.add_argument("-s", "--seed", help="pseudo-random number seed", type=int)
    solve_parser.add_argument("-o", "--output", help="output filename")
    solve_parser.add_argument("--problem_module", help="module containing the problem (if not built-in)")
    solve_parser.add_argument("--algorithm_module", help="module containing the algorithm (if not built-in)")
    solve_parser.add_argument("arguments", metavar="KEY=VALUE", nargs="*", help="additional arguments to set")

    filter_parser = subparsers.add_parser("filter", help="filter results using selected filters")
    filter_parser.add_argument("-e", "--epsilons", help="epsilon values for epsilon-dominance, implies --nondominated", type=split_list(float))
    filter_parser.add_argument("-f", "--feasible", help="remove any infeasible solutions", action='store_true')
    filter_parser.add_argument("-u", "--unique", help="remove any duplicate solutions", action='store_true')
    filter_parser.add_argument("-n", "--nondominated", help="remove any dominated solutions", action='store_true')
    filter_parser.add_argument("-o", "--output", help="output filename")
    filter_parser.add_argument("filename", help="input filename", nargs="?")

    normalize_parser = subparsers.add_parser("normalize", help="normalize results")
    normalize_parser.add_argument("-r", "--reference_set", help="reference set")
    normalize_parser.add_argument("--minimum", help="minimum values for each objective", type=split_list(float))
    normalize_parser.add_argument("--maximum", help="maximum values for each objective", type=split_list(float))
    normalize_parser.add_argument("-o", "--output", help="output filename")
    normalize_parser.add_argument("filename", help="input filename", nargs="?")

    plot_parser = subparsers.add_parser("plot", help="generate simple 2D or 3D plot")
    plot_parser.add_argument("-t", "--title", help="plot title")
    plot_parser.add_argument("-o", "--output", help="output filename")
    plot_parser.add_argument("filename", help="input filename", nargs="?")

    args = parser.parse_args(input)

    if args.log:
        logging.basicConfig(level=args.log)

    debug_inputs(args)

    if args.command == "hypervolume":
        ref_set = load_set(args.reference_set)
        input_set = load_set(args.filename)
        hyp = platypus.Hypervolume(reference_set=ref_set)
        print(hyp.calculate(input_set))
    elif args.command == "gd":
        ref_set = load_set(args.reference_set)
        input_set = load_set(args.filename)
        gd = platypus.GenerationalDistance(reference_set=ref_set)
        print(gd.calculate(input_set))
    elif args.command == "igd":
        ref_set = load_set(args.reference_set)
        input_set = load_set(args.filename)
        igd = platypus.InvertedGenerationalDistance(reference_set=ref_set)
        print(igd.calculate(input_set))
    elif args.command == "epsilon":
        ref_set = load_set(args.reference_set)
        input_set = load_set(args.filename)
        eps = platypus.EpsilonIndicator(reference_set=ref_set)
        print(eps.calculate(input_set))
    elif args.command == "spacing":
        input_set = load_set(args.filename)
        spacing = platypus.Spacing()
        print(spacing.calculate(input_set))
    elif args.command == "solve":
        if args.seed is not None:
            random.seed(args.seed)

        problem_module = importlib.import_module(coalesce(args.problem_module, "platypus"))
        algorithm_module = importlib.import_module(coalesce(args.algorithm_module, "platypus"))

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
        problem_args = type_cast(only_keys_for(extra_args, problem_class), problem_class)
        algorithm_args = type_cast(only_keys_for(extra_args, algorithm_class), algorithm_class)

        log_args(problem_args, problem_class.__name__)
        log_args(algorithm_args, algorithm_class.__name__)

        problem = problem_class(**problem_args)
        algorithm = algorithm_class(problem, **algorithm_args)
        algorithm.run(args.nfe)

        save_set(algorithm, args.output)
    elif args.command == "filter":
        input_set = load_set(args.filename)

        if args.unique:
            input_set = platypus.unique(input_set)
        if args.feasible:
            input_set = platypus.feasible(input_set)
        if args.nondominated or args.epsilons:
            archive = platypus.EpsilonBoxArchive(args.epsilons) if args.epsilons else platypus.Archive()
            archive += input_set
            input_set = archive

        save_set(list(input_set), args.output)
    elif args.command == "normalize":
        input_set = load_set(args.filename)
        minimum = args.minimum
        maximum = args.maximum

        if args.reference_set:
            if minimum is not None or maximum is not None:
                LOGGER.warn("ignoring --minimum and --maximum options since a reference set is provided", file=sys.stderr)
            ref_set = load_set(args.reference_set)
            minimum, maximum = platypus.normalize(ref_set)

        norm_min, norm_max = platypus.normalize(input_set, minimum, maximum)
        LOGGER.info(f"Using bounds minimum={norm_min}; maximum={norm_max}")

        for s in input_set:
            s.objectives[:] = s.normalized_objectives

        save_set(input_set, args.output)
    elif args.command == "plot":
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

        ax.set_title(coalesce(args.title, args.filename))
        ax.set_xlabel("$f_1(x)$")
        ax.set_ylabel("$f_2(x)$")

        if nobjs == 3:
            ax.set_zlabel("$f_3(x)$")

        if args.output:
            plt.savefig(args.output)
        else:
            plt.show()

if __name__ == "__main__":
    main()
