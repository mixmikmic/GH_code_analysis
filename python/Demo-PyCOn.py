def json(f):
    def wrapper(*args):
        
        return 'Bolaji'
    return wrapper

def noun(name):
    def sample(f):
        def wrapper(*args):
            return "{} said {}".format(name, f(*args))
        return wrapper
    return sample

@json
def add(a,b):
    return a + b

@noun('Bolaji')
def say_pycon():
    return 'PyCon Nigeria 2017'

add(5,2)

say_pycon()

add(70,23)

a=1

a

a=2

a



