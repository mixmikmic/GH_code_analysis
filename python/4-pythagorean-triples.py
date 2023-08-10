from itertools import count, islice
from math import sqrt
from functools import reduce

# The Pythagorean triples with c <= 100.
test_cases = (
    (3, 4, 5),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
    (20, 21, 29),
    (12, 35, 37),
    (9, 40, 41),
    (28, 45, 53),
    (11, 60, 61),
    (16, 63, 65),
    (33, 56, 65),
    (48, 55, 73),
    (13, 84, 85),
    (36, 77, 85),
    (39, 80, 89),
    (65, 72, 97),
)

def test():
    for known_good, unknown in zip(test_cases, pythagorean_triples()):
        assert known_good == unknown

def gcd(a, b):
    while True:
        remainder = a % b
        if remainder == 0:
            return b
        a, b = b, remainder

def gcd(a, b):
    while True:
        a, b = b, a % b
        if b == 0:
            return a

def gcd(a, b):
    while True:
        c = a % b
        if c == 0:
            return b
        a = b % c
        if a == 0:
            return c
        b = c % a
        if b == 0:
            return a

def pythagorean_triples():
    for c in count():
        c_squared = c * c
        for a in range(1, c):
            a_squared = a*a
            b_squared = c_squared - a_squared
            if a_squared > b_squared:
                break
            b = int(sqrt(b_squared))
            if a_squared + b*b == c_squared and reduce(gcd, (a, b, c)) == 1:
                yield a, b, c

test()


list(islice(pythagorean_triples(), 16))

