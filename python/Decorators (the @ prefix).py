def factorial(n):
    
    return 1 if n == 0 else n*factorial(n-1)

import time

def timer(fnc):
    
    def inner(arg):
        
        t0 = time.time()
        fnc(arg)
        t1 = time.time()
        return t1-t0
    
    return inner


timed_factorial = timer(factorial)
timed_factorial(500)

@timer
def timed_factorial(n):
    
    return 1 if n == 0 else n*factorial(n-1)


timed_factorial(500)

