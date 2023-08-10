def nth_triangle(n):
    return int(n*(n+1)/2)

nth_triangle(7)

import math

def n_factors(n):
    factors = 0
    
    for i in range(1,int(math.sqrt(n))):
        if n % i == 0:
            factors += 1
    
    return factors*2

n_factors(28)

def first_tri_n_factors(n):
    i = 1
    while True:
        tri = nth_triangle(i)
        n_fac = n_factors(tri)
        if n_fac>n:
            return tri
        i += 1

first_tri_n_factors(500)

