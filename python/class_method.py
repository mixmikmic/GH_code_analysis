class A(object):
    v = {'a': 3}
    @classmethod
    def f(cls):
        print 'In A'

class B(object):
    v = 5
    @classmethod
    def f(cls):
        print 'In B'

a = A()
a.__class__.f()

B.f()

b = B()
b.f()

a.f()

print A.v
print B.v

a.v

b.v

a.__class__.v

x=3







