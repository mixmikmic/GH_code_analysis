class MyClass:
    """A simple example class"""
    i = 12345

    def f(self):
        return 'hello world'

x = MyClass()

x.__doc__    # the doc attribute

x.i    # valid attribute of type data

x.f    # valid attribute of type method

x.__class__   # return the class of any object

x.f.__name__    # get the name attribute of the method

mystring = 'test'
mystring.__class__

x.f()   

x.counter = 1
x.counter

class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

x = Complex(3.0, -4.5)   # instantiate and pass arguments
x.r, x.i

class Dog:
    legs = 4    #  this is a class variable
    
    def __init__(self, name):
        self.name = name
        self.tricks = []    #  instance variable

    def add_trick(self, trick):
        self.tricks.append(trick)

d = Dog('Fido')
d.add_trick('roll over')
d.tricks

d.legs



