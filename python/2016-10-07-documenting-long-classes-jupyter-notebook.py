from __future__ import print_function
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod

class AbstractBase(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def foo(self, x):
        pass
    
    @abstractmethod
    def bar(self, y):
        pass

class A(AbstractBase):
    def foo(self, x):
        return x

try:
    a = A()
except TypeError as ex:
    print(ex)

class A(A):
    def bar(self, y):
        return y ** 2

a = A()

a.bar(2)

class B(object):
    pass

print(B)

print(id(B))
C = B
print(id(C))

class B(object):
    pass

print(id(B))

print(id(C))
print(C)

class D(object):
    pass

print(id(D))

class D(D):
    pass

print(id(D))
print([id(base_class) for base_class in D.__bases__])

class E(object):
    x = 'foo'
    
class E(E):
    y = 'bar'
    
print(E.x, E.y)

get_ipython().system('jupyter nbconvert --to html --template jekyll.tpl 2016-10-07-documenting-long-classes-jupyter-notebook')

