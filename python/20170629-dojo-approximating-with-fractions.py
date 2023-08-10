from math import pi

from fractions import Fraction

pi

pi_fraction = Fraction(3141592653589793, 1000000000000000)
float(pi_fraction)

def get_approximations(x, type_=None, n=None):
    """yields (numerator, denominator) tuples
    that approximate x with increasing accuracy.
    Up to n approximations are yielded.
    n defaults to 10.
    type_ defaults to float."""
    x = abs(x)
    if n is None:
        n = 10
    if type_ is None:
        type_ = float
    numerators = [0, 1]
    denominators = [1, 0]
    for _ in range(n):
        # It is fun to play with continued fractions.
        integer, fraction = divmod(x, 1)
        integer = int(integer)
        numerators.append(integer*numerators[-1] + numerators[-2])
        denominators.append(integer*denominators[-1] + denominators[-2])
        yield numerators[-1], denominators[-1]
        if fraction == 0:
            break
        x = type_(1) / fraction

def show_float_approximations(x, n=None):
    for i, (numerator, denominator) in enumerate(get_approximations(x, float, n), 1):
        error = numerator / denominator - x
        error_fraction = Fraction(numerator, denominator) - Fraction(x)
        print(f'{i} {numerator}/{denominator} ({100*error:.3}% error) ({float(100*error_fraction):.3}% error)')

x = pi
show_float_approximations(x)

def show_fraction_approximations(x, n=None):
    print(repr(x))
    for i, (numerator, denominator) in enumerate(get_approximations(x, Fraction, n), 1):
        error = Fraction(numerator, denominator) - x
        print(f'{i} {numerator}/{denominator} ({float(100*error):.3}% error)')

# Python floats have limited precision.

x = pi_fraction
show_float_approximations(x, 20)

# Python fractions have arbitrary precision.

x = pi_fraction
show_fraction_approximations(x, 20)

# let's try some more digits

pi_long = '3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'
i = pi_long.rindex('.') 
pi_just_digits = pi_long[:i] + pi_long[i+1:]
pi_fraction = Fraction(int(pi_just_digits), 10**(len(pi_long) - (i+1)))
pi_fraction

x = pi_fraction
show_float_approximations(x, 300)

x = pi_fraction
show_fraction_approximations(x, 300)

