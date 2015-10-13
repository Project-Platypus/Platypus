
class Algorithm(object):
    
    def __init__(self, problem):
        super(Algorithm, self).__init__()
        self.problem = problem
        self.nfe = 0
    
    def step(self):
        raise NotImplementedError("Method not implemented")
    
    def run(self, NFE):
        start_nfe = self.nfe
        
        while self.nfe - start_nfe < NFE:
            self.step()
    
class NSGAII(Algorithm):
    
    def __init__(self, problem):
        super(NSGAII, self).__init__(problem)
        
    def step(self):
        