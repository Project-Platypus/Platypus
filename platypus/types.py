# Copyright 2015 David Hadka
#
# This file is part of Platypus.
#
# Platypus is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Platypus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Platypus.  If not, see <http://www.gnu.org/licenses/>.
from abc import ABCMeta

class Type(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Type, self).__init__()
        
class Real(Type):
    
    def __init__(self, min_value, max_value):
        super(Real, self).__init__()
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        
    def __str__(self):
        return "Real(%f, %f)" % (self.min_value, self.max_value)
        
class Binary(Type):
    
    def __init__(self, nbits):
        super(Binary, self).__init__()
        self.nbits = nbits
        
    def __str__(self):
        return "Binary(%d)" % self.nbits
    
class Permutation(Type):
    
    def __init__(self, elements):
        super(Permutation, self).__init__()
        self.elements = elements
        
    def __str__(self):
        return "Permutation"
        