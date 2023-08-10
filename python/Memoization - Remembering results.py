def prime(n):
    
    for i in range(n, 0, -1):
        if all([i // x != i / x for x in range(i-1, 1, -1)]):
            return i
        
print(prime(1000000))

cache = {}

def cached_prime(n):
    
    if n in cache:
        return cache[n]    
    for i in range(n, 0, -1):
        if all([i // x != i / x for x in range(i-1, 1, -1)]):
            cache[n] = i
            return i
        
print(cached_prime(1000000))
print(cached_prime(1000000))

def memoize(fnc):
    
    cache = {}
    
    def inner(*args):
        
        if args in cache:
            return cache[args]
        cache[args] = fnc(*args)
        return cache[args]
    
    return inner


@memoize
def memoized_prime(n):
    
    for i in range(n, 0, -1):
        if all([i // x != i / x for x in range(i-1, 1, -1)]):
            return i
        

print(memoized_prime(1000000))
print(memoized_prime(1000000))

