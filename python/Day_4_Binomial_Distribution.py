import math

def input():
    return "1.09 1"

ratios = [float(i) for i in input().strip().split(" ")]
probs = {"b": ratios[0] /sum(ratios), "g": ratios[1] / sum(ratios)}
probs

def binomial_dist(x, n, p):
    q = 1.0 - p
    bernoulli = p**x * q**(n-x)
    combination = math.factorial(n) / (math.factorial(x) * math.factorial(n-x))
    return combination * bernoulli

[(i, binomial_dist(4, 10, 0.5)) for i in range(3, 15)]

print("{0:.3f}".format(sum([binomial_dist(i, 6, probs["b"]) for i in range(3, 7)])))



