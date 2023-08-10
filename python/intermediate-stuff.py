class A(object):
    def __foo(self):
        print "A foo"
    def class_(self):
        self.__foo()
        print self.__foo.__name__
    def __doo__(self):
        print "doo"
        
a = A()
print hasattr(a, '__foo') # where has this gone?
a.class_()
a.__doo__()

print dir(a)

a._A__foo()

class P(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return P(self.x + other.x, self.y + other.y)
    
    def __gt__(self, other):
        #if both x and y component is greater than the other object's x and y
        return (self.x > other.x) and (self.y > other.y)
    
    def __str__(self):
        return "x : %s, y : %s" % (self.x, self.y)
    
p1 = P(0,0)
p2 = P(3,4)
p3 = P(1,3)

print p3 + p2
print p1 > p2
print p2 > p1

class Seq(object):
    def __getitem__(self, i):
        if type(i) is slice:
            # this has edge case issues, but just a demo!
            return list(range(i.start, i.stop))
        else:
            return i

s = Seq()
print s[5]
print s[-4]
print s[2:5]

l = [i for i in range(0,5)]
l2 = [i*i for i in range(0,5)]
print l
print l2

l = [i for i in range(0,5) if i % 2 ==0]
print l

# get all combinations where x > y and x, y < 5
xy = [ (x, y) for x in range (0,5) for y in range (0, 5) if x > y]
print xy

# we can even call functions
l = [x.upper() for x in "hello"]
print l

# creating lists of lists is also a synch
gre = "hello how are you doing?"
[[s.lower(), s.upper(), len(s)] for s in gre.split()]

# nested comprehensions - we can do it, but it may not be very readable
matrix = [[i+x for i in range(3)] for x in range(3)]
print matrix

# we can also have a comprehension for dicts
d = {x : x**2 for x in range(5)}
print d

# lambda is used when you need anonymous functions defined as an expression
# in this example you could define a function and pass it to foo, or use the lambda
# in this case the lambda is neater.
# lambdas can take in n number of params, and the body is a single expression that is also the return value

def foo(list_, func):
    l = []
    for i in list_:
        l.append(func(i))
    return l

def sq(i):
    return i**2

l = [i for i in range(5)]
print foo(l, sq)
print foo(l, lambda x : x**2)

class P(object):
    def __init__(self, x):
        self.x = x
    def __str__(self):
        return "x : %s" % self.x

l = [P(5), P(2), P(1), P(4), P(3)]
l.sort(cmp=lambda x, y: x.x - y.x)
for p in l : print p # [str(p) for p in l]
    
# there are many more complex and cryptic ways to use (exploit) lambdas, 
#     you can search for it online if you are interested
# check lambda with multiple args
# lambda *x : sys.stdout.write(" ".join(map(str, x)))

# filter is a function that takes an interable and a callable 
#  applies the function to each element,i.e. ret = func(element)
#  and returns a list with elements for which 'ret' was true

l = range(0,10)
l = filter(lambda x : x%2==0, l)
print l, type(l)

# zip is to sew together a bunch of iterables
# the list generated is of the minimum size of all the iterators that have gone in!

a = [1,2,3,4,5]
b = (0,4,6,7)
c = {1:'a', 7:'b', 'm':'v'}

print zip(a,b,c)

# map - takes in a iterable and callable - applies the callable to each element of the iterable
#  returns a new list with each element being the return value of "callable(elem)"

print map(lambda x: x**2, range(10))

# map is extremely useful as a shorthand for "applying" a function across an iterable,
#  especially in conjunction with lambda
import sys
my_print = lambda *x : sys.stdout.write(" ".join(map(str, x)))
my_print("hello", "how are you", 1234)

# 1. Iterators

class myitr(object):
    def __init__(self, ulimit=5):
        self.limit = ulimit
    def __iter__(self):
        self.index = 0
        return self
    def next(self):
        if self.index < self.limit:
            self.index += 1
            return self.index
        else:
            raise StopIteration
            
itr = myitr()
for i in itr:
    print i

# 2. Generators

def gen(lim):
    i = 0
    while i < lim:
        yield i
        i = i + 1
        
for i in gen(5):
    print i

# 3. Generator expression

def seq(num):
    return (i**2 for i in range(num))

for i in seq(5):
    print i

# 4. Overriding __getitem__

class Itr(object):
    def __init__(self, x):
        self.x = x
        
    def __getitem__(self, index):
        if index < self.x:
            return index
        else:
            raise StopIteration


for i in Itr(5):
    print i

# closure example - raised_to_power returns a fn that takes a variable and raises to the power 'n'
#  'n' is passed only once - while defining the function!

def raised_to_power(n):
    def fn(x):
        return x**n
    return fn
    
p2 = raised_to_power(2)
p3 = raised_to_power(3)

print p2(2), p2(3) # still remembers that n=2
print p3(2), p3(3) # still remembers that n=3

# have to be cautious!

def power_list(n):
    '''returns list of fn, each raises to power i, where i : 0 --> n'''
    fn_list = []

    def fn(x):
        return x**i
    
    for i in range(n):
        # doesn't matter if fn was defined here either
        fn_list.append(fn)

    return fn_list

for j in power_list(4):
    print j(2) # prints 2 power 3, 4 times
    

# decorator is just a nicer way of defining a closure - more syntactic sugar

def deco(fn):
    def new_fn(*args, **kwargs):
        print "entring function", fn.__name__
        ret = fn(*args, **kwargs)
        print "exiting function", fn.__name__
    return new_fn

@deco
def foo(x):
    print "x : ", x
    
foo(4)

# Another example

def add_h1(fn):
    def nf(pram):
        return "<h1> " + fn(pram) + " </h1>"
    return nf

@add_h1
def greet(name):
    return "Hello {0}!".format(name)

print greet("Nutanix")

# decorator that takes parameter

def add_h(num):
    def deco(fn):
        # this is the decorator for a specific 'h'
        def nf(pram):
            return "<h%s> "%num + fn(pram) + " </h%s>"%num
        return nf
    return deco

@add_h(3)
def greet(name):
    return "Hello {0}!".format(name)
print greet("Nutanix")


# we can have multiple decorators as well
@add_h(2)
@add_h(4)
def greet2(name):
    return "Hello {0}!".format(name)

print greet2("Nutanix")
        

class A(object):
    def __init__(self):
        print "A.init"        
    def foo(self):
        print "A.foo"
        
class B(A):
    def __init__(self):
        print "B.init"
    def foo(self):
        print "B.foo"

class C(A):
    def __init__(self):
        print "C.init"
    def foo(self):
        print "C.foo"
        
class D(B, C):
    def __init__(self):
        print "D.init"
    #def foo(self):
    #    print "D.foo"
    
class E(C, B):
    def __init__(self):
        print "E.init"

d = D()
d.foo() 

e = E()
e.foo()

# we see that fn lookup's happen in the order of declaration of parent in the child's definition.

class A(object):
    def __init__(self):
        print "A.init"        
    def foo(self):
        print "A.foo"
        
class B(A):
    def __init__(self):
        print "B.init"
    def foo(self):
        print "B.foo"

class C(A):
    def __init__(self):
        print "C.init"
    def foo(self):
        print "C.foo"
        
class D(C):
    def __init__(self):
        print "D.init"
    def foo(self):
        print "D.foo"
    
class E(D, C): # you can't have (C, D) - TypeError: Cannot create a consistent MRO
    def __init__(self):
        print "E.init"

e = E()
e.foo()
E.__mro__

# so what's mro - (explain in live session)

class A(object):
    def __init__(self, x):
        self.x = x
    def __getattr__(self, val):
        print "getattr val :", val, type(val)
        return val

a = A(3)
print "X :", a.x # getattr not called for x
ret = a.y
print "Y :", ret

class A(object):
    def __init__(self, x):
        self.x = x
    def __getattr__(self, val):
        print "getattr"
        return val
    def __setattr__(self, name, val):
        print "setattr"
        if name == 'x':
            self.__dict__[name] = val

a = A(3)
print a.x
print a.y

# setattr is called for both
a.y = 5
a.x = 5

class MulBy(object):
    def __init__(self, x):
        self.x = x
    def __call__(self, n):
        print "here!"
        return self.x * n
    
m = MulBy(5)
print m(3)

class X(object):
    def __new__(cls, *args, **kwargs):
        print "new"
        print args, kwargs
        return object.__new__(cls)
        
    def __init__(self, *args, **kwargs):
        print "init"
        print args, kwargs
        
x = X(1,2,3,a=4)

class WindowsVM(object):
    def __init__(self, state="off"):
        print "New windows vm. state : %s" %state
    def operation(self):
        print "windows ops"
        
class LinuxVM(object):
    def __init__(self, state="off"):
        print "New linux vm. state : %s" %state
    def operation(self):
        print "linux ops"

class VM(object):
    MAP = {"Linux" : LinuxVM, "Windows": WindowsVM}
    
    def __new__(self, vm_type, state="off"):
        # return object.__new__(VM.MAP[vm_type]) #--doesn't call init of other class
        vm = object.__new__(VM.MAP[vm_type])
        vm.__init__(state)
        return vm


vm1 = VM("Linux")
print type(vm1)
vm1.operation()
print ""
vm2 = VM("Windows", state="on")
print type(vm2)
vm2.operation()

# simple example 
class C(object):
    def __init__(self):
        self._x = None

    def getx(self):
        print "getx"
        return self._x
    
    def setx(self, value):
        print "setx"
        self._x = value
        
    def delx(self):
        print "delx"
        del self._x
        
    x = property(getx, setx, delx, "I'm the 'x' property.")
    
c = C()
c.x = 5 # so when we use 'x' variable of a C object, the getters and setters are being called!
print c.x
del c.x

print C.x

# the same properties can be used in form of decorators!
class M(object):
    def __init__(self):
        self._x = None

    @property
    def x(self):
        print "getx"
        return self._x
    
    @x.setter
    def x(self, value):
        print "setx"
        self._x = value
    
    @x.deleter
    def x(self):
        print "delx"
        del self._x

m = C()
m.x = 5 
print m.x
del m.x

# This is a pure python implementation of property

class Property(object):
    "Emulate PyProperty_Type() in Objects/descrobject.c"

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
    
# during the live session, explain how this maps to the previous decorator version of property.

class MyMet(type):
    """Here we see that MyMet doesn't inherit 'object' but rather 'type' class - the builtin metaclass
    """
    def __new__(cls, name, bases, attrs):
        """
        Args:
          name (str) : name of the new class being created
          bases (tuple) : tuple of the classes which are the parents of cls
          attrs (dict) : the attributes that belong to the class
        """
        print "In new"
        print name
        print bases
        print attrs
        return super(MyMet, cls).__new__(cls, name, bases, attrs)
        
    def __init__(self, *args, **kwargs):
        print "In init"
        print self
        print args
        print kwargs
        
class Me(object):
    __metaclass__ = MyMet
    
    def foo(self):
        print "I'm foo"
    

m = Me()
m.foo()

class MyMet(type):
    """Here we see that MyMet doesn't inherit 'object' but rather 'type' class - the builtin metaclass
    """
    def __new__(cls, name, bases, attrs):
        """
        Args:
          name (str) : name of the new class being created
          bases (tuple) : tuple of the classes which are the parents of cls
          attrs (dict) : the attributes that belong to the class
        """
        print "In new"
        print name
        print bases
        print attrs
        def foo(self):
            print "I'm foo"
        attrs['foo'] = foo
        return super(MyMet, cls).__new__(cls, name, bases, attrs)
        
    def __init__(self, name, bases, attrs):
        print "In init"
        print self # actually the object being created
        print name
        print bases
        print attrs
        def bar(self):
            print "I'm bar"
        setattr(self, "bar", bar)
        
    def test(self):
        print "in test"
        
    #def __call__(self):
    #    print "self :", self
    # Note : If I override call here, then I have to explicitly call self.__new__
    #        otherwise it is completely skipped. Normally a class calls type's __call__
    #        which re-routes it to __new__ of the class
         
        
class Me(object):
    __metaclass__ = MyMet




print "\n-------------------------------\n"

m = Me()
print type(Me) # not of type 'type' anymore!
m.foo()
m.bar()
# print m.test --attribute error
Me.test()

class A(object):
    def __init__(self):
        print "init A"
    def foo(self):
        print "foo A"
    def bar(self):
        print "bar A"
        
class B(object):
    def __init__(self):
        print "init B"
    def doo(self):
        print "doo B"
    def bar(self):
        print "bar B"
        
def test(self):
    print "Self : ", self

Cls = type("C", (A,B), {"test": test})

c = Cls()
print Cls
print Cls.__name__, type(Cls)
print c

c.foo()
c.bar()
c.doo()
c.test()



