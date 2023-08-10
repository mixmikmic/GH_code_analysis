def factorial(n):
    
    return 1 if n == 0 else n*factorial(n-1)

import time


def timer_with_arguments(units='s'):

    def timer(fnc):

        def inner(arg):

            """Inner function"""

            t0 = time.time()
            fnc(arg)
            t1 = time.time()        
            diff = t1-t0
            if units == 'ms':
                diff *= 1000
            return diff

        return inner
    
    return timer


timed_factorial = timer_with_arguments(units='ms')(factorial)
timed_factorial(500)

@timer_with_arguments(units='s')
def factorial(n):
    
    return 1 if n == 0 else n*factorial(n-1)


factorial(100)

