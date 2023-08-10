from math import sqrt


def p_pythagoras(x, y):
    
    return sqrt(x**2 + y**2)

p_pythagoras(1, 1)

l_pythagoras = lambda x, y: sqrt(x**2 + y**2)
l_pythagoras(1,1)

def f_factorial(n):
    
    return 1 if n == 0 else n*f_factorial(n-1)


f_factorial(3)

l_factorial = lambda n: 1 if n == 0 else n*l_factorial(n-1)
l_factorial(3)

l = [0, 1, 2, 3, 4]
list(map(lambda x: x*2, l))

