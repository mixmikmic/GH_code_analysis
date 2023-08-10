l_factorial = lambda n: 1 if n == 0 else n*l_factorial(n-1)

def chain_mul(*what):
    
    """Takes a list of (function, argument) tuples. Calls each
    function with its argument, multiplies up the return values,
    (starting at 1) and returns the total."""
    
    total = 1
    for (fnc, arg) in what:
        total *= fnc(arg)
    return total


chain_mul( (l_factorial, 2), (l_factorial, 3) )

import operator


def chain(how, *what):
        
    total = 1
    for (fnc, arg) in what:
        total = how(total, fnc(arg))
    return total


chain(operator.truediv, (l_factorial, 2), (l_factorial, 3) )

