from functools import wraps
def debug_on(func):
    @wraps(func) #preserving the metadata for func
    def debugging(*args,**kwargs):
        retval = func(*args,**kwargs);
        print('Scope: debugging %s:%s:%s'%(func,func.__name__,retval));
        return retval;
    print('Scope: debug_on',debugging)
    return debugging

# the @ syntax is equivalent to function nesting https://www.python.org/dev/peps/pep-0318/
# debug_on(sum)

@debug_on
def sum(x,y):        # sum->debugging
    '''Return the sum'''
    return x+y;
@debug_on
def sum2(x,y):       #sum2->debugging, but seperate instance
    '''Return the square of the sum'''
    return (x+y)**2.;

print('Scope: main sum %s, sum2 %s'%(sum,sum2))
print('-'*10)
print('Q:What does %s do? A:%s'%(sum2.__name__,sum2.__doc__))

sum(3,4)
sum2(3,4)

@debug_on
class point():
    @debug_on
    def __init__(self,x,y):
        self.x=x;
        self.y=y;
    @debug_on
    def dist(self):
        return self.x**2.+self.y**2;

a = point(1,2);
a.dist()

def debug_on_classes(cls):
    @wraps(cls) #preserving the cls metadata
    def debugging(*args,**kwargs):
        cdict = cls.__dict__;
        for item in cdict:
            func = getattr(cls,item);
            if( hasattr(func,'__call__') ):
                setattr(cls,item,debug_on(func))
        return cls(*args,**kwargs);
    return debugging;

@debug_on_classes
class point():
    def __init__(self,x,y):
        self.x=x;
        self.y=y;
    def dist(self):
        return self.x**2.+self.y**2;
a = point(1,2);
a.dist();

from timeit import default_timer as timer
from time import sleep;

def time_execution(*args,**kwargs):
    symb = kwargs.pop('symb','*'*10)
    def time_decorator(func):
        def time_function(*args,**kwargs):
            start = timer();
            result=func(*args,**kwargs);
            end = timer()-start;
            print('%s Function call %s took %.2f seconds'%(symb,func.__name__,end));
            return result;
        return time_function;
    return time_decorator;

@time_execution(symb='='*10)               # note that @profile is provided by https://mg.pov.lt/profilehooks/
def recursive_counter(n=0):
    while(n<10):
        sleep(0.1);
        n = recursive_counter(n+1);
    return n;

print( recursive_counter(0) )

