import platypus
import functools


def belegundu(vars):
    
    x = vars[0]
    y = vars[1]
    
    return [-2*x + y, 2*x + y], [-x + y - 1, x + y - 7]


def generation_objectives(algorithm_obj, repository):
    
    repository.append(platypus.unique(algorithm_obj.result))
        
    return repository


def objectives_repository(
    population_size,
    n_generation,
    feasibility=False, 
    nondominated=False
):
    
    # problem object
    problem = platypus.Problem(2, 2, 2)
    problem.types[:] = [platypus.Real(0, 5), platypus.Real(0, 3)]
    problem.constraints[:] = "<=0"
    problem.function = belegundu
    
    # repository for generation wise objectives
    algorithm = platypus.NSGAII(
        problem, 
        population_size=population_size
    )
    repository=[]
    algorithm.run(
        population_size * n_generation, 
        callback=functools.partial(generation_objectives, repository=repository)
    )
    
    # only feasible solutions if `feasibility` is True
    if feasibility:
        repository = map(lambda solutions: [i for i in solutions if i.feasible], repository)
    else:
        pass
    
    # only non-dominated solutions if `nondominance` is True
    if nondominated:
        repository = map(lambda solutions: platypus.nondominated(solutions), repository)
    else:
        pass
    
    return list(repository)

output = objectives_repository(
    population_size=10, 
    n_generation=10, # number of generation 
)

print(output)