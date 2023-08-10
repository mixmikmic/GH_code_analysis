def camelcase(s):
    
    return ''.join([w.capitalize() for w in s.split('_')])

print(camelcase('some_function'))

class Just:
    
    def __init__(self, value):
        
        self._value = value
        
    def bind(self, fnc):
        
        try:
            return Just(fnc(self._value))
        except:
            return Nothing()
    
    def __repr__(self):
        
        return self._value
    


class Nothing:
    
    def bind(self, fnc):
        
        return Nothing()
    
    def __repr__(self):
        
        return 'Nothing'
    
    
print(Just('some_function').bind(camelcase))
print(Nothing().bind(camelcase))
print(Just(10).bind(camelcase))

class List:
    
    def __init__(self, values):
        
        self._values = values
        
    def bind(self, fnc):
        
        return List([fnc(value) for value in self._values])
    
    def __repr__(self):
        
        return str(self._values)
    
    
List(['some_text', 'more_text']).bind(camelcase)

