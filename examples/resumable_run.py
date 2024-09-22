from platypus import NSGAII, DTLZ2, save_state, load_state

try:
    import jsonpickle
except ImportError:
    print("Please install jsonpickle to run this example!")

problem = DTLZ2()

algorithm = NSGAII(problem)
algorithm.run(5000)

# Save the algorithm state to a file.
save_state("state.json", algorithm, json=True, indent=4)

# Load the state and continue running.
algorithm = load_state("state.json", json=True)
algorithm.run(5000)
