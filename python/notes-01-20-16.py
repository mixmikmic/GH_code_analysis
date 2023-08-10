N = 10
fib_list = [1,1]
for n in range(2,N+1):
    next_fib = fib_list[n-1] + fib_list[n-2]
    fib_list.append( next_fib )
print(fib_list)

def factorial(n):
    "Compute the factorial n! of a positive integer."
    product = 1
    if n == 0 or n == 1:
        return product
    else:
        for d in range(2,n+1):
            product = product * d
        return product

print(factorial(5))

def factorial(n):
    "Compute the factorial n! of a positive integer."
    product = 1
    if n == 0 or n == 1:
        return product
    else:
        for d in range(2,n+1):
            product = product * d
        return product

def e_approx(N):
    "Compute the Nth partial sum of the Taylor series of e^x centered at 0 evaluated at x=1."
    terms_in_series = []
    for n in range(0,N+1):
        # We can use our factorial function defined above to compute the terms of the series
        terms_in_series.append(1 / factorial(n))
    return sum(terms_in_series)

print(e_approx(3))
print(e_approx(10))

def factorial(n):
    "Compute the factorial n! of a positive integer."
    product = 1
    if n == 0 or n == 1:
        return product
    else:
        for d in range(2,n+1):
            product = product * d
        return product

def e_approx_2(N):
    "Compute the Nth partial sum of the Taylor series of e^x centered at 0 evaluated at x=0."
    series_sum = 0
    for n in range(0,N+1):
        series_sum = series_sum + 1 / factorial(n)
    return series_sum

print(e_approx_2(3))
print(e_approx_2(10))

def pythagorean(N):
    "Find all Pythagorean triples [a,b,c] with c less than or equal to N."
    
    # Create an empty list so that we can append Pythagorean triples to the list when we find them
    py_triples = []
        
    # Loop over all possible values of a and b
    # We can restrict to 1 <= a <= b < N
    # Loop over values of b up to N
    for b in range(1,N):
        
        # Loop over values of a up to b
        for a in range(1,b+1):
            
            # Test if a^2 + b^2 is equal to a square c^2 with c <= N, and append if True
            c = round( (a ** 2 + b ** 2) ** 0.5 )
            if (a ** 2 + b ** 2 == c ** 2) and (c <= N):
                py_triples.append([a,b,c])
                
    return py_triples

print(pythagorean(10))

print(pythagorean(30))

len(pythagorean(500))

for triple in pythagorean(85):
    if triple[2] == 85:
        print(triple)

pythagorean(85)[-10:]

list_of_c = []
for triple in pythagorean(50):
    if triple[2] not in list_of_c:
        list_of_c.append(triple[2])
for n in range(1,51):
    if n not in list_of_c:
        print(n,'is not in the list.')

