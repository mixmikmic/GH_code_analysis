def factorial(num):
    res = 1
    while num > 0:
        res *= num
        num -= 1
    return res

def n_choose_k(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))

def negative_binomial(x, n, p):
    return n_choose_k(n - 1, x - 1) * (p**x) * ((1 - p)**(n - x))

# p * (1-p)^(n-1)
def geometric(n, p):
    return round(negative_binomial(1, n, p), 3)

geometric(5, 1/3)



