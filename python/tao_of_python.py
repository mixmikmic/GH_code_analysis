two = 2
print(type(two))

print(type(type(two)))

print(type(two).__bases__)

print(dir(two))

class A:
    pass
a = A()
print(type(a))
print(type(A))
print(A.__bases__)

A = type('A', (), {})
a = A()
print(type(a))
print(type(A))
print(A.__bases__)
print(isinstance(a, A), isinstance(a, object), issubclass(A, object))

def f():
    """My name is f."""
    pass
print(type(f))
print(type(type(f)))
print(type(f).__bases__)
print(f.__doc__)

issubclass(type, object) # Recap rule #1

issubclass(object, object) # Recap rule #1

issubclass(object, type) # Recap rule #1

isinstance(object, type) # Recap rule #2

isinstance(type, type) # Recap rule #2

isinstance(type, object) # Recap rule #3

isinstance(object, object) # Recap rule #3

from IPython.display import Image, display
display(Image(url='figures/mind_blown.gif', width=400))

