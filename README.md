# Platypus-Optimize

**Note: Platypus is current under development and has not been extensively
tested.  Please refrain from using in applications until a stable release is
available.**

### What is Platypus-Optimize?

Platypus is a framework for evolutionary computing in Python with a focus on
multiobjective evolutionary algorithms (MOEAs).  There are a number of Python
libraries for optimization, including PyGMO, Inspyred, DEAP and Scipy, but only
Platypus provides extensive support for multiobjective optimization.

1. **Minimal Setup** - With only minimal information about the optimization
   problem, Platypus automatically fills in the rest.  You can always specify
   more details, but Platypus will automatically supply missing options based
   on best practices.
   
2. **Pure Python** - Unlike other libraries including PyGMO, Inspyred, DEAP, and
   Scipy, Platypus is focused on minimizing dependencies on non-pure Python
   libraries such as Numpy (Numpy is great, but can be challenging to install
   in certain environments).  By eliminating these dependencies, Platypus can
   be used on any system where Python is available.
   
3. **Parallelization** - Platypus is designed from the bottom up with
   parallelization in mind, both using local threading and distributed
   computing across a network.
   
4. **Compatibility with Python Ecosystem** - Python supports many powerful
   modeling and analysis frameworks, and Platypus is designed with these in
   mind.  Integration into tools such as OpenMDAO are built-in.

For example, optimizing a simple biobjective problem with a single real-valued
decision variables is accomplished in Platypus with:

```python
    from platypus.core import Problem
    from platypus.types import Real
    from platypus.algorithms import NSGAII
    
    def schaffer(x):
    	return [x**2, (x-2)**2]

    problem = Problem(1, 2)
    problem.variables[:] = Real(0, 1)
    problem.objectives[:] = Problem.MINIMIZE
    problem.function = schaffer

    algorithm = NSGAII(problem)
    algorithm.run(10000)

    for solution in algorithm.result:
        print solution
```
