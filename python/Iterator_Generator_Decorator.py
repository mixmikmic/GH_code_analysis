class Counter:
    def __init__(self,start,end):
        self.start = start
        self.end = end
    def __iter__(self):
        return self
    def __next__(self):
        if self.start != self.end:
            self.start += 1
            return self.start - 1
        else:
            raise StopIteration()
        
c = Counter(1,10)
list(c)

class CountGen:
    def __init__(self,start,end):
        self.start = start
        self.end = end
    def __iter__(self):
        while self.start != self.end:
            yield self.start
            self.start += 1
            
c = CountGen(1,10)
list(c)

class ReverseStringIterator:
    def __init__(self,string):
        self.string = string
        self.idx  = len(string)
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx == 0:
            raise StopIteration()
        else:
            self.idx -= 1
            return self.string[self.idx]
    
s = ReverseStringIterator('SPARTA Is This')
for x in s:
    print(x,end='')

class ReverseStringGenerator:
    def __init__(self,string):
        self.string = string
    def __iter__(self):
        for x in reversed(self.string):
            yield x
s = ReverseStringGenerator('SoapOpera')
''.join(list(s))

def deco(func):
    def wrapp(*args,**kwargs):
        print('-----------')
        res = func(*args,**kwargs)
        name = func.__name__
        print('Name ->',name)
        print('Args -> ',*args,**kwargs)
        print('Res ->',res)
        print('------------')
    return wrapp

@deco
def sq(x):
    return x*x

sq(10)

from functools import wraps
def debug(prefix=''):
    def decorate(func):
        msg = prefix+'-> '+func.__qualname__
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(msg)
            return func(*args,**kwargs)
        return wrapper
    return decorate

@debug(prefix='sparta')
def sq(x):
    return x*x

sq(10)

#Fn for decorate
def mydecofun(fn):
    def wrapper(*args,**kwargs):
        print('My deco for -> ',fn.__qualname__)
        return fn(*args,**kwargs)
    return wrapper

#Apply decorate function for all methods in class
def debugallmethods(cls):
    #vars(cls) gives dictornay filled with all attributes
    for name,val in vars(cls).items():
        #check if val is callable
        if callable(val):
            #setattr(obj,name,value)
            setattr(cls,name,mydecofun(val))
    return cls



@debugallmethods
class Spam:
    def __init__(self):
        self.x = 10
    def print(self):
        print(self.x)
        
s = Spam()
s.print()

#Captures all get attrubutes class
def debugattributes(cls):
    orig_getattribute = cls.__getattribute__
    
    def __getattribute__(self,name):
        print('Geting : ',name)
        return orig_getattribute(self,name)
    cls.__getattribute__=__getattribute__
    return cls

@debugattributes
class Spam:
    def __init__(self):
        self.x = 10
        self.y = 20
    def printall(self):
        print(self.x)
        print(self.y)
        
s = Spam()
s.printall()



