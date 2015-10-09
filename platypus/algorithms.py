
class Algorithm(object):
    
    def __init__(self, problem):
        super(Algorithm, self).__init__()
        self.problem = problem
        self.nfe = 0
    
    def step(self):
        raise NotImplementedError("Method not implemented")