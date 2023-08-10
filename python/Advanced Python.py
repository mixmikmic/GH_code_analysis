import functools

# decorator function, which does basic logging
def log(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kw):
        print("Before decoration")
        result = fun(*args, **kw)
        print("After invocation")
        return result
    return wrapper

# base function
@log
def sum(a, b):
    return a + b

# equivalent to
#sum = log(sum)

sum(1,3)

def test(n, l=[]):
    for i in range(n):
        l.append(i)
    print(l)

test(3)
test(3, [1,2,3])
test(3)

# it can be noticed that the mutable argument changes
# are memorized in the method signature
import inspect
inspect.signature(test)



