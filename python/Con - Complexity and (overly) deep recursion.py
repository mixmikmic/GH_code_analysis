def p_factorial(n):
    
    f = 1
    for i in range(1, n+1):
        f *= i
    return f


print(p_factorial(0)) # = 1 by convention
print(p_factorial(2)) # = 1×2 = 2
print(p_factorial(4)) # = 1×2×3x4 = 24

def f_factorial(n):
    
    return 1 if n == 0 else n*f_factorial(n-1)


print(f_factorial(0)) # = 1 by convention
print(f_factorial(2)) # = 1×2 = 2
print(f_factorial(4)) # = 1×2×3x4 = 24

f_factorial(1000)

