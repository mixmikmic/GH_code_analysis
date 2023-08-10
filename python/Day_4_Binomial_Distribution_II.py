import math

def input():
    return "12 10"

inputs = [float(i) for i in input().strip().split(" ")]
rejects_prob = inputs[0] / 100
trial = inputs[1]
rejects, trial

def binomial_dist(x, n, p):
    q = 1.0 - p
    bernoulli = p**x * q ** (n-x)
    combination = math.factorial(n) / (math.factorial(x) * math.factorial(n-x))
    return combination * bernoulli

at_least_2 = sum(binomial_dist(i, trial, rejects_prob) for i in range(2, 11))

at_most_2 = sum(binomial_dist(i, trial, rejects_prob) for i in range(1, 3))

at_least_2, at_most_2

[(i, round(binomial_dist(i, 100, .12), 6)) for i in range(1, 20)]



