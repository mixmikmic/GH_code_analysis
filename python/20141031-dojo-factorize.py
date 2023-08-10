def factorize(x):
    if x <= 1 or divmod(x, 1)[1] != 0:
        raise ValueError
    factors = []
    f = 2
    while f * f <= x:
        if x % f == 0:
            factors.append(f)
            x //= f
        else:
            f += 1
    if x > 1:
        factors.append(x)
    return factors

n = 52
factorize(n)

known_good_output = factorize(n)

def largest_factor(x):
    return factorize(x)[-1]

largest_factor(n)

largest_factor(int('1'*4))

def test_largest_factor_52():
    assert(largest_factor(52) == 13)

n = 52
get_ipython().magic('timeit factorize(n)')

def factorize(x):
    if x <= 1 or divmod(x, 1)[1] != 0:
        raise ValueError
    factors = []
    f = 2
    while f * f <= x:
        if x % f == 0:
            factors.append(f)
            x //= f
        else:
            if f > 2:
                f += 2
            else:
                f = 3
    if x > 1:
        factors.append(x)
    return factors

n = 52
assert known_good_output == factorize(n)
get_ipython().magic('timeit factorize(n)')

1.25 % 1, -1.25 % 1, 2 % 1

import math

def foo(n):
    get_ipython().magic('timeit int(n) != n')
    get_ipython().magic('timeit n % 1 != 0')
    get_ipython().magic('timeit math.modf(n)[0] != 0')
    get_ipython().magic('timeit divmod(n, 1)[1] != 0')
    get_ipython().magic('timeit n != math.trunc(n)')
    get_ipython().magic('timeit n != math.floor(n)')
    get_ipython().magic('timeit n != math.ceil(n)')

foo(1.5)

foo(1234)

def factorize(x):
    if x <= 1 or x != int(x):
        raise ValueError
    factors = []
    f = 2
    while f * f <= x:
        if x % f == 0:
            factors.append(f)
            x //= f
        else:
            if f > 2:
                f += 2
            else:
                f = 3
    if x > 1:
        factors.append(x)
    return factors

n = 52
assert known_good_output == factorize(n)
get_ipython().magic('timeit factorize(n)')

def factorize(x):
    if x <= 1 or x != int(x):
        raise ValueError
    factors = []
    f = 2
    while x > 1:
        if x % f == 0:
            factors.append(f)
            x //= f
        else:
            f += 1
    return factors

n = 52
assert known_good_output == factorize(n)
get_ipython().magic('timeit factorize(n)')

def factorize(x):
    if x <= 1 or x != int(x):
        raise ValueError
    factors = []
    f = 2
    while x > 1:
        if x % f == 0:
            factors.append(f)
            x //= f
        else:
            if f == 2:
                f = 3
            else:
                f += 2
    return factors

n = 52
assert known_good_output == factorize(n)
get_ipython().magic('timeit factorize(n)')

def factorize(x):
    if x <= 1 or x != int(x):
        raise ValueError
    factors = []
    f = 2
    while x > 1:
        while x % f == 0:
            factors.append(f)
            x //= f
        else:
            if f == 2:
                f = 3
            else:
                f += 2
    return factors

n = 52
assert known_good_output == factorize(n)
get_ipython().magic('timeit factorize(n)')

