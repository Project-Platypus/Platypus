# Platypus

[![PyPI Latest Release](https://img.shields.io/pypi/v/Platypus-Opt.svg)](https://pypi.org/project/Platypus-Opt/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Platypus-opt)
![PyPI - Status](https://img.shields.io/pypi/status/Platypus-opt)


[![Test and Publish](https://github.com/Project-Platypus/Platypus/actions/workflows/test-and-publish.yml/badge.svg)](https://github.com/Project-Platypus/Platypus/actions/workflows/test-and-publish.yml) [![Documentation Status](https://readthedocs.org/projects/platypus/badge/?version=latest)](http://platypus.readthedocs.org/en/latest/?badge=latest)
![GitHub last commit](https://img.shields.io/github/last-commit/Project-Platypus/Platypus)


[![PyPI](https://img.shields.io/pypi/dm/Platypus-Opt.svg)](https://pypi.org/project/Platypus-Opt/)
![GitHub Repo stars](https://img.shields.io/github/stars/Project-Platypus/Platypus)
![GitHub forks](https://img.shields.io/github/forks/Project-Platypus/Platypus)
![GitHub License](https://img.shields.io/github/license/Project-Platypus/Platypus)


### What is Platypus?

Platypus is a framework for evolutionary computing in Python with a focus on
multiobjective evolutionary algorithms (MOEAs).  It differs from existing
optimization libraries, including PyGMO, Inspyred, DEAP, and Scipy, by providing
optimization algorithms and analysis tools for multiobjective optimization.
It currently supports NSGA-II, NSGA-III, MOEA/D, IBEA, Epsilon-MOEA, SPEA2, GDE3,
OMOPSO, SMPSO, and Epsilon-NSGA-II.  For more information, see our
[examples](examples/)
and [online documentation](http://platypus.readthedocs.org/en/latest/index.html).

### Example

For example, optimizing a simple biobjective problem with a single real-valued
decision variables is accomplished in Platypus with:

```python

    from platypus import NSGAII, Problem, Real

    def schaffer(x):
        return [x[0]**2, (x[0]-2)**2]

    problem = Problem(1, 2)
    problem.types[:] = Real(-10, 10)
    problem.function = schaffer

    algorithm = NSGAII(problem)
    algorithm.run(10000)
```

### Installation

To install the latest Platypus release, run the following command:

```
    pip install platypus-opt
```

To install the latest development version of Platypus, run the following commands:

```
    pip install -U build setuptools
    git clone https://github.com/Project-Platypus/Platypus.git
    cd Platypus
    python -m build
    python -m pip install --editable .
```

#### Anaconda

Platypus is also available via conda-forge.

```
    conda config --add channels conda-forge
    conda install platypus-opt
```

For more information, see the [feedstock](https://github.com/conda-forge/platypus-opt-feedstock).

### Citation

If you use this software in your work, please cite it as follows (APA style):

> Hadka, D. (2024). Platypus: A Framework for Evolutionary Computing in Python (Version 1.4.1) [Computer software].  Retrieved from https<span>://</span>github.com/Project-Platypus/Platypus.

### License

Platypus is released under the GNU General Public License.
