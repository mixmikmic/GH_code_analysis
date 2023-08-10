[ n**2 for n in range(1,11) ]

N = 252
[ divisor for divisor in range(1,N+1) if N % divisor == 0 ]

[ [a,b,c] for a in range(1,4) for b in range(1,4) for c in range(1,4) if (a != b and b != c and a != c) ]

[ [a,b,a**2+b**2] for b in range(1,6) for a in range(1,b+1) ]

[ [ divisor for divisor in range(1,n+1) if n % divisor == 0 ] for n in range(1,21) ]

import number_theory as nt

nt.is_prime(16193)

nt.primes_up_to(70)

nt.primes_interval(2000,2100)

nt.twin_primes_interval(10,43)

nt.twin_primes_interval(3000,3200)

nt.prime_divisors(100)

dir(nt)

get_ipython().magic('pinfo nt.is_prime')

