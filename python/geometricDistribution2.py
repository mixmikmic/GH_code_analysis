def factorial(num):
    res = 1
    while num > 0:
        res *= num
        num -= 1
    return res

def n_choose_k(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))

def binomial(x, n, p):
    return n_choose_k(n, x) * (p**x) * ((1 - p)**(n - x))

def negative_binomial(x, n, p):
    return binomial(x - 1, n - 1, p) * p

# p * (1-p)^(n-1)
def geometric(n, p):
    return round(negative_binomial(1, n, p), 3)

def proba_at_least_one_success_n_trials(n, p):
    return round(sum([geometric(n_ + 1, 1/3) for n_ in range(n)]), 3)

assert geometric(5, 1/3) == 0.066

assert proba_at_least_one_success_n_trials(5, 1/3) == 0.868

def quick_proba_at_least_one_success_n_trials(n, p):
    return round(1 - (1 - p)**n, 3)

assert quick_proba_at_least_one_success_n_trials(5, 1/3) == 0.868



