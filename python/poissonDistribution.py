import math

def factorial(num):
    if num == 0:
        return 1
    res = num
    while num > 1:
        num -= 1
        res *= num
    return res

def poisson(avg_rate, k):
    return (avg_rate**k) * math.exp(-avg_rate) / factorial(k)

avg_rate, k = 2, 3
assert round(poisson(avg_rate, k), 3) == 0.18

avg_rate = 5
assert round(sum([poisson(avg_rate, k) for k in range(4)]), 3) == 0.265



