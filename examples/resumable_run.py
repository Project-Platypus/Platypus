from platypus import NSGAII, DTLZ2, save_state, load_state

# Example of saving and resuming a run using a state file.  By default, the
# state is stored in a binary format using pickling.  Setting json=True
# will save to a readable JSON format.

problem = DTLZ2()

algorithm = NSGAII(problem)
algorithm.run(5000)

# Save the algorithm state to a file.
save_state("NSGAII_DTLZ2_State.bin", algorithm)

# Load the state and continue running.
algorithm = load_state("NSGAII_DTLZ2_State.bin")
algorithm.run(5000)

print("Total NFE:", algorithm.nfe)
