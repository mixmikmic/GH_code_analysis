def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 1
    if n > 1:
        factors.append(n)
    return factors

prime_factors(2*2)

n = 600851475143
get_ipython().magic('timeit prime_factors(n)')
prime_factors(n)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        if i > 2:
            i += 2
        else:
            i = 3
    if n > 1:
        factors.append(n)
    return factors

n = 600851475143
get_ipython().magic('timeit prime_factors(n)')
prime_factors(n)

from math import sqrt

def prime_factors(n):
    i = 2
    factors = []
    sqrt_n = int(sqrt(n))
    while i <= sqrt_n:
        while n % i == 0:
            factors.append(i)
            n //= i
        sqrt_n = int(sqrt(n))
        if i > 2:
            i += 2
        else:
            i = 3
    if n > 1:
        factors.append(n)
    return factors

n = 600851475143
get_ipython().magic('timeit prime_factors(n)')
prime_factors(n)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while True:
            quotient, remainder = divmod(n, i)
            if remainder != 0:
                break
            factors.append(i)
            n = quotient
        if i > 2:
            i += 2
        else:
            i = 3
    if n > 1:
        factors.append(n)
    return factors

n = 600851475143
get_ipython().magic('timeit prime_factors(n)')
prime_factors(n)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while True:
            quotient, remainder = divmod(n, i)
            if remainder != 0:
                break
            factors.append(i)
            n = quotient
        if i > 2:
            i += 2
        else:
            i = 3
    if n > 1:
        factors.append(n)
    return factors

n = 600851475143
get_ipython().magic('timeit prime_factors(n)')
prime_factors(n)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += (2 if i > 2 else 1)
    if n > 1:
        factors.append(n)
    return factors

n = 600851475143
get_ipython().magic('timeit prime_factors(n)')
prime_factors(n)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += (i > 2) + 1
    if n > 1:
        factors.append(n)
    return factors

n = 600851475143
get_ipython().magic('timeit prime_factors(n)')
prime_factors(n)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += (i > 2)
        i += 1
    if n > 1:
        factors.append(n)
    return factors

n = 600851475143
get_ipython().magic('timeit prime_factors(n)')
prime_factors(n)

