def sum_sieve(limit):
    sum_primes = 0
    primes = [True] * limit
    primes[0], primes[1] = False, False
    for p in range(2, limit):
        if primes[p]:
            sum_primes += p
            for i in range(2*p, limit, p):
                primes[i] = False
    return sum_primes

sum_sieve(2_000_000)

