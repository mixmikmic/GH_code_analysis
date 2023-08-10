def factorial(n):
    if n == 1:
        return n
    else: 
        return n * factorial(n-1)

factorial(10)

factorial(100)

x = list(str(factorial(100)))

sum([int(i) for i in x])

