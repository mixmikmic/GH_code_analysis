def is_prime(n):
    "Determine if n is a prime number."
    for d in range(2,n):
        if n % d == 0:
            # n is divisible by d
            return False 
    # n is not divisible by any d
    return True

for n in range(2,50):
    if is_prime(n):
        print(n,"is prime!")

def primes_up_to(N):
    "Compute list of primes p =< N."
    primes = []
    for n in range(2,N+1):
        if is_prime(n):
            primes.append(n)
    return primes

for N in range(50,60):
    print(N,':',primes_up_to(N))

def divisors(N):
    "Compute the list of divisors of N."
    divisors_list = [1]
    for d in range(2,N):
        if N % d == 0:
            divisors_list.append(d)
    divisors_list.append(N)
    return divisors_list

divisors(10)

divisors(2048)

def divisor_sum(N,k):
    "Compute the k power sum of divisors of N."
    return sum([d**k for d in divisors(N)])

divisor_sum(5,1)

