def test(required, *args, **kwargs): 
    print (required)
    if args: 
        print (args)
    if kwargs: 
        print (kwargs)

test()

test('sup world')

test('sup world', 'this is an arg')

test('sup world', 'arg1', 2, 'arg3')

test('sup world', 'arg1', key1 = 'keyword', key2 = 100, d=[53])

class Car: 
    def __init__(self, color, mileage): 
        self.color = color 
        self.mileage = mileage
        
class AlwaysBlueCar(Car): 
    '''
    - subclass of Car that always has Blue color property
    - use the super().__init__ method to inherit all properties of Car parent
    - use args and kwargs to avoid writing __init__ args 
    '''
    def __init__(self, *args, **kwargs): 
        super().__init__(self, *args, **kwargs) 
        #override color 
        self.color = 'blue'



