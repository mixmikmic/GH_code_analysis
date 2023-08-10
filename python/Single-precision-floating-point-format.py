from resources.utils import run_tests

def binary_to_decimal(bit_vector):
    assert len(bit_vector) == 32
    
    fraction = bit_vector[:23]
    exponent = bit_vector[23:31]
    sign = bit_vector[-1]
    
    value_sign = (-1) ** sign
    e = sum([val * (2 ** i)
             for i, val in enumerate(exponent)])
    value_exponent = 2 ** (e - 127)
    value_fraction = 1 + sum([
        fraction[len(fraction) - i] * (2 ** (-i))
        for i in range(1, len(fraction) + 1)
    ])
    return value_sign * value_fraction * value_exponent

sign = [0]
exponent = [0, 1, 1, 1, 1, 1, 0, 0]
fraction = ([0, 1] + [0] * 21)
bit_vector = (sign + exponent + fraction)[::-1]
binary_to_decimal(bit_vector)

tests = [
    (dict(bit_vector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]), 0.15625),
    (dict(bit_vector=([0, 0] + [1] * 7 + [0] * 23)[::-1]), 1),
    (dict(bit_vector=([1, 1] + [0] * (7 + 23))[::-1]), -2),
    (dict(bit_vector=([0] + [1] * 8 + [0] * 23)[::-1]), 3.402823669209385e+38)
]

run_tests(tests, binary_to_decimal)

# TODO: decimal to binary

