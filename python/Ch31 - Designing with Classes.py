class C(object):
    def method(self, x):
        print(x)
    def method(self, x, y):
        print(x, y)
        
c = C()
c.method(1)

class Wrapper(object):
    def __init__(self, obj):
        self.wrapped = obj
    def __getattr__(self, attrname):
        print('Trace: '+attrname)
        # getattr(x, n) works like x.__dict__[n] except that it does an inheritnace search
        return getattr(self.wrapped, attrname)
    
x = Wrapper([1, 2, 3])
x.append(4)

x.wrapped

class C(object):
    def __print_value(self, x):
        print(x)

c = C()
c.__print_value(1)

# These attribute can still be referenced

c._C__print_value(1)

class Spam(object):
    def doit(self, message):
        print(message)

obj = Spam()
t = Spam.doit         # a pure function in Python3
t(obj, 'Unbound')

obj = Spam()
x = obj.doit           # instance + function
x('Bounded')  

class Selfless(object):
    def __init__(self, data):
        self.data = data
    def selfless(arg1, arg2):
        return arg1 + arg2
    def normal(self, arg1, arg2):
        return self.data + arg1 + arg2
    
x = Selfless(2)

x.normal(1, 2)

Selfless.selfless(1, 2)

def factory(aClass, *pargs, **kargs):
    return aClass(*pargs, **kargs)

class Spam(object):
    def doit(self, message):
        print(message)
    
class Person:
    def __init__(self, name, job=None):
        self.name = name
        self.job = job
        
obj1 = factory(Spam)
obj2 = factory(Person, 'Bob')
print(obj1)
print(obj2)

