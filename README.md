# Platypus

### What is Platypus?

Platypus is a framework for evolutionary computing in Python with a focus on
multiobjective evolutionary algorithms (MOEAs).

1. **Minimal Setup** - With only minimal information about the optimization
   problem, Platypus automatically fills in the rest.  You can always specify
   more details, but Platypus will automatically supply missing options based
   on best practices.
   
2. **Parallelization** - Platypus is designed from the bottom up with
   parallelization in mind.
   
3. **Compatability with Python Ecosystem** - Python supports many powerful
   modeling and analysis frameworks, and Platypus is designed with these in
   mind.  Integration into tools such as OpenMDAO are built-in.

For example, optimizing a simple biobjective problem with a single real-valued
decision variables is accomplished in Platypus with:

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

        
        
    from platypus.parallel import IslandModel
    from platypus.core import Problem
    from platypus.types import Real
    from platypus.algorithms import NSGAII
    
    def schaffer(x):
    	return [x**2, (x-2)**2]

    problem = Problem(1, 2)
    problem.variables[:] = Real(0, 1)
    problem.objectives[:] = Problem.MINIMIZE
    problem.function = schaffer
    
    cluster = IslandModel(NSGAII, problem, islands=4)
    cluster.migration_frequency = 100
    cluster.topology = IslandModel.FULL
    cluster.run(100000)

    for solution in cluster.result:
        print solution
    