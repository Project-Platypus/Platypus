from platypus import NSGAII, DTLZ2, FixedFrequencyExtension, save_json

class SaveResults(FixedFrequencyExtension):

    # Called at the specified frequency (every 100 NFE)
    def do_action(self, algorithm):
        save_json(f"NSGAII_DTLZ2_{algorithm.nfe}.json", algorithm.result)

    def end_run(self, algorithm):
        save_json("NSGAII_DTLZ2_Final.json", algorithm.result)


problem = DTLZ2()

algorithm = NSGAII(problem)
algorithm.add_extension(SaveResults(frequency=100))
algorithm.run(10000)
