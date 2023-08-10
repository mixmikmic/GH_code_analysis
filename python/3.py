def largest_prime(n):
    largest = 0
    x = 2
    while (x**2 <= n):
        if (n % x == 0):
            n /= x
            largest = x
        else:
            x += 1
            
    if n > largest:
        largest = n
    
    return largest

largest_prime(600851475143)

