import sympy as sym
import pandas as pd

def goldbach(N):
    """Returns all pairs of primes that sum to give N"""
    primes = list(sym.primerange(1, N))
    sums = []
    for i, p1 in enumerate(primes):
        for p2 in primes[i:]:
            if p1 + p2 == N:
                sums.append((p1, p2))
    return sums

maxN = 500
data = [[N, *pair] for N in range(4, maxN + 1) 
        for pair in goldbach(N) if N % 2 == 0 ]

df = pd.DataFrame(data, columns=["N","a", "b"])  # Create a data frame
df.to_excel("data/goldbach.xlsx")  # Write it to excel

