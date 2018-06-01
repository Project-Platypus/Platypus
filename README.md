# Platypus

[![Build Status](https://travis-ci.org/Project-Platypus/Platypus.svg?branch=master)](https://travis-ci.org/Project-Platypus/Platypus)
[![Documentation Status](https://readthedocs.org/projects/platypus/badge/?version=latest)](http://platypus.readthedocs.org/en/latest/?badge=latest)

### What is Platypus?

Platypus is a framework for evolutionary computing in Python with a focus on
multiobjective evolutionary algorithms (MOEAs).  It differs from existing
optimization libraries, including PyGMO, Inspyred, DEAP, and Scipy, by providing
optimization algorithms and analysis tools for multiobjective optimization.
It currently supports NSGA-II, NSGA-III, MOEA/D, IBEA, Epsilon-MOEA, SPEA2, GDE3,
OMOPSO, SMPSO, and Epsilon-NSGA-II.  For more information, see our
[IPython Notebook](https://gist.github.com/dhadka/ba6d3c570400bdb411c3)
or our [online documentation](http://platypus.readthedocs.org/en/latest/index.html).

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
    git clone https://github.com/Project-Platypus/Platypus.git
    cd Platypus
    python setup.py install
```

#### Anaconda

Platypus is also available via conda-forge. 

```
    conda config --add channels conda-forge
    conda install platypus-opt
```

For more information see the [feedstock](https://github.com/conda-forge/platypus-opt-feedstock) located here. 

### License

Platypus is released under the GNU General Public License.
