import math

def is_prime(x):
    if x <= 1:
        return False
    elif x == 2:
        return True
    elif x & 1 == 0:
        return False
    elif x == 3 or x == 5 or x == 7:
        return True
    elif x % 3 == 0 or x % 5 == 0:
        return False
    
    isqrt = int(math.sqrt(x))
    
    for i in range(2, isqrt + 1):
        if x % i == 0:
            return False
    
    return True

def is_prime_gap(x):
    if x == 1:
        return True
    elif x & 1 == 1:
        return False
    
    level = 1
    while True:
        margins = is_prime(level) and is_prime(level + x)
        if not margins:
            level += 1
        else:
            clean = True
            for n in range(level + 1, level + x):
                if is_prime(n):
                    clean = False
                    break
            if clean:
                print (level, level + x)
                return True
            else:
                level += 1
                
for i in range(2, 101, 2):
    print i, is_prime_gap(i)
            

