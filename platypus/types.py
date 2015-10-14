from abc import ABCMeta

class Type(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Type, self).__init__()
        self.value = None
        
class Real(Type):
    
    def __init__(self, min_value, max_value):
        super(Real, self).__init__()
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        
    def __str__(self):
        return "Real(%f, %f)" % (self.min_value, self.max_value)
        
class Int(Type):
    
    def __init__(self, min_value, max_value):
        super(Int, self).__init__()
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        
    def __str__(self):
        return "Int(%d, %d)" % (self.min_value, self.max_value)
        