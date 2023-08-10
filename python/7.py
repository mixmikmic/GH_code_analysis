import numpy as np

def sieve(target, limit):
    primes = [True] * limit
    primes[0], primes[1] = False, False
    for p in range(2, limit):
        if primes[p]:
            for i in range(2*p, limit, p):
                primes[i] = False
    count = 0
    for i, p in enumerate(primes):
        if p == True:
            count += 1
        if count == target:
            return i

sieve(10001, 125_000)

