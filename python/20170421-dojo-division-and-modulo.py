def show_math(numerator, divisor):
    quotient = numerator // divisor
    remainder = numerator % divisor
    assert (quotient, remainder) == divmod(numerator, divisor)
    print(numerator, divisor, quotient, remainder)

# just integers

divisor = 3
for numerator in range(-2*divisor-1, 2*divisor + 1):
    show_math(numerator, divisor)

# floats and integers

divisor = 3
numerators = map(
    float,
    '-3.1 -3 -2.9 -2.1 -2 -1.9 -1.1 -1 -0.9 -0.1 0 .1 .9 1 1.1 1.9 2 2.9 3 3.1'.split())
for numerator in numerators:
    show_math(numerator, divisor)

# just floats

from math import e

divisor = e
for numerator in (x/2 for x in range(-6, 6+1)):
    show_math(numerator, divisor)

