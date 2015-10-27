# Platypus

**Note: Platypus is current under development and has not been extensively
tested.  Please refrain from using in applications until a stable release is
available.**

[![Build Status](https://travis-ci.org/Project-Platypus/Platypus.svg?branch=master)](https://travis-ci.org/Project-Platypus/Platypus)
[![Documentation Status](https://readthedocs.org/projects/platypus/badge/?version=latest)](http://platypus.readthedocs.org/en/latest/?badge=latest)

### What is Platypus?

Platypus is a framework for evolutionary computing in Python with a focus on
multiobjective evolutionary algorithms (MOEAs).  It differs from existing
optimization libraries, including PyGMO, Inspyred, DEAP, and Scipy, by providing
optimization algorithms and analysis tools for multiobjective optimization.
It currently supports the following algorithms:

| Algorithm    | Original Authors               
| -------------|----------------------------------------------------------------- |
| NSGA-II      | K. Deb, A. Pratap, S. Agarwal, T. Meyarivan                      |
| NSGA-III     | K. Deb, H. Jain                                                  |
| MOEA/D       | H. Li, Q. Zhang                                                  |
| IBEA         | E. Zitzler, S. Kunzli                                            |
| Epsilon MOEA | K. Deb, M. Mohan, S. Mishra                                      |
| SPEA2        | E. Zitzler, M. Laumanns, L. Thiele                               |
| GDE3         | S. Kukkonen, J. Lampinen                                         |
| OMOPSO       | M. R. Sierra, C. A. Coello Coello                                |
| SMPSO        | A. J. Nebro, J. J. Durillo, J. Garcia-Nieto, C. A. Coello Coello |

### Design Goals

Platypus is currently under development and we welcome new collaborators.
The design of Platypus is focused on the following:

1. **Multiobjective** - Focus on solving multiobjective optimization problems.

2. **Separation of Concerns** - There should be a clear separation between
   the problem definition and the method of solving the problem.  Doing so
   allows swapping in different algorithms or operators without altering the
   problem formulation.
   
3. **Minimal Setup** - Minimize the amount of code needed to define and
   optimize a problem.  Platypus automatically supplies missing options based
   on best practices.
   
4. **Pure Python** - Minimize external dependencies on non-pure Python
   libraries.  By eliminating such dependencies, Platypus can run on any system
   where vanilla Python is installed.
   
5. **Cloud Computing** - Scientific computing with multiobjective evolutionary
   algorithms is computationally expensive.  Platypus should facilitate
   distributed computing with minimal setup.
   
6. **Ecosystem** - Python supports many powerful modeling, design and analysis
   frameworks (e.g., OpenMDAO), and Platypus should facilitate collaboration
   with these tools.

### Example

For example, optimizing a simple biobjective problem with a single real-valued
decision variables is accomplished in Platypus with:

```python
    from platypus.core import Problem
    from platypus.types import Real
    from platypus.algorithms import NSGAII
    
    def schaffer(x):
    	return [x**2, (x-2)**2]

    problem = Problem(1, 2)
    problem.types[:] = Real(0, 1)
    problem.function = schaffer

    algorithm = NSGAII(problem)
    algorithm.run(10000)

    for solution in algorithm.result:
        print solution
```

### License

Platypus is released under the GNU General Public License.
