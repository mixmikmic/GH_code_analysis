from abc import ABCMeta, abstractmethod

class Vehicle(metaclass=ABCMeta):
    @abstractmethod
    def change_gear(self):
        pass

    @abstractmethod
    def start_engine(self):
        pass

class Car(Vehicle): # subclass the ABC, abstract methods MUST be overridden
    def __init__(self, make, model, color):
        self.make = make
        self.model = model
        self.color = color

try:
    car = Car('Toyota', 'Avensis', 'silver')
except TypeError as e:
    print(e)

class ObjectCreator:
    pass
print(ObjectCreator)  # you can pass a class as a parameter because it's an object

print(hasattr(ObjectCreator, 'new_attribute'))
ObjectCreator.new_attribute = 'foo' # you can add attributes to a class
print(hasattr(ObjectCreator, 'new_attribute'))
print(ObjectCreator.new_attribute)

ObjectCreatorMirror = ObjectCreator # you can assign a class to a variable
print(ObjectCreatorMirror.new_attribute)
print(ObjectCreatorMirror())

import re

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# the metaclass will automatically get passed the same arguments that we pass to `type`
def camel_to_snake_case(name, bases, attrs):
    """Return a class object, with its attributes from camelCase to snake_case."""
    print("Calling the metaclass camel_to_snake_case to construct class: {}".format(name))
    
    # pick up any attribute that doesn't start with '__' and snakecase it
    snake_attrs = {}
    for attr_name, attr_val in attrs.items():
        if not attr_name.startswith('__'):
            snake_attrs[convert(attr_name)] = attr_val
        else:
            snake_attrs[attr_name] = attr_val
    return type(name, bases, snake_attrs) # let `type` do the class creation

class MyVector(metaclass=camel_to_snake_case):
    def addToVector(self, other): pass
    def subtractFromVector(self, other): pass
    def calculateDotProduct(self, other): pass
    def calculateCrossProduct(self, other): pass
    def calculateTripleProduct(self, other): pass

print([a for a in dir(MyVector) if not a.startswith('__')])

def meta_function(name, bases, attrs):
    print('Calling meta_function')
    return type(name, bases, attrs)


class MyClass1(metaclass=meta_function):
    def __new__(cls, *args, **kwargs):
        """
        Called to create a new instance of class `cls`. __new__ takes the class
        of which an instance was requested as its first argument. The remaining 
        arguments are those passed to the object constructor expression 
        (the call to the class). The return value of __new__ should be the 
        new object instance (usually an instance of cls).
        """
        print('MyClass1.__new__({}, *{}, **{})'.format(cls, args, kwargs))
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """
        Called after the instance has been created (by __new__), but before it 
        is returned to the caller. The arguments are those passed to the object
        constructor. Note: both __new__ and __init__ receive the same arguments.
        """
        print('MyClass1.__init__({}, *{}, **{})'.format(self, args, kwargs))

a = MyClass1(1, 2, 3, x='ex', y='why')

class MyMeta(type):
    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        """
        Called before the class body is executed and it must return a dictionary-like object
        that's used as the local namespace for all the code from the class body.
        """
        print("Meta.__prepare__(mcs={}, name={}, bases={}, **{}".format(
            mcs, name, bases, kwargs))
        return {}

    def __new__(mcs, name, bases, attrs, **kwargs):
        """
        Like __new__ in regular classes, which returns an instance object of the class
        __new__ in metaclasses returns a class object, i.e. an instance of the metaclass
        """
        print("MyMeta.__new__(mcs={}, name={}, bases={}, attrs={}, **{}".format(
            mcs, name, bases, list(attrs.keys()), kwargs))
        return super().__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs, **kwargs):
        """
        Like __init__ in regular classes, which initializes the instance object of the class
        __init__ in metaclasses initializes the class object, i.e. the instance of the metaclass
        """
        print("MyMeta.__init__(cls={}, name={}, bases={}, attrs={}, **{}".format(
            cls, name, bases, list(attrs.keys()), kwargs))
        super().__init__(name, bases, attrs)

        # Note: all three above methods receive as arguments:
        # 1. The name, bases and attrs of the future class that will be created
        # 2. Keyword arguments passed in the class inheritance list

    def __call__(cls, *args, **kwargs):
        """
        This is called when we make an instance of the class constructed with the metaclass
        """
        print("MyMeta.__call__(cls={}, args={}, kwargs={}".format(cls, args, kwargs))
        self = super().__call__(*args, **kwargs)
        print("MyMeta.__call__ return: ", self)
        return (self)

print("Metaclass MyMeta created")

class MyClass2(metaclass=MyMeta, extra=1):
    def __new__(cls, s, a=0, b=0):
        print("MyClass2.__new__(cls={}, s={}, a={}, b={})".format(cls, s, a, b))
        return super().__new__(cls)

    def __init__(self, s, a=0, b=0):
        print("MyClass2.__init__(self={}, s={}, a={}, b={})".format(self, s, a, b))
        self.a, self.b = a, b

print("Class MyClass created")

a = MyClass2('hello', a=1, b=2)
print("MyClass instance created: ", a)

class CamelToSnake(type): 
    def __new__(mcs, name, bases, attrs):
        # pick up any attribute that doesn't start with '__' and snakecase it
        snake_attrs = {}
        for attr_name, attr_val in attrs.items():
            if not name.startswith('__'):
                snake_attrs[convert(attr_name)] = attr_val
            else:
                snake_attrs[attr_name] = attr_val
        return super().__new__(mcs, name, bases, snake_attrs)

class MyVector(metaclass=CamelToSnake):
    def addToVector(self, other): pass
    def subtractFromVector(self, other): pass
    def calculateDotProduct(self, other): pass
    def calculateCrossProduct(self, other): pass
    def calculateTripleProduct(self, other): pass

print([a for a in dir(MyVector) if not a.startswith('__')])

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class SnakeSingleton(CamelToSnake, Singleton):
    pass

class MyVector(metaclass=SnakeSingleton):
    def addToVector(self, other): pass
    def subtractFromVector(self, other): pass
    def calculateDotProduct(self, other): pass
    def calculateCrossProduct(self, other): pass
    def calculateTripleProduct(self, other): pass

print([a for a in dir(MyVector) if not a.startswith('__')])
v1 = MyVector(); v2 = MyVector()
print(v1 is v2)

