
class Type(object):
    
    def __init__(self):
        super(Type, self).__init__()
        self.value = None
        
    def __getitem__(self, key):
        print(key)
        return None
        
class Real(Type):
    
    def __init__(self, min_value, max_value):
        super(Real, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        
class Int(Type):
    
    def __init__(self, min_value, max_value):
        super(Int, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        