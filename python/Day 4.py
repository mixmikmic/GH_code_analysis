15 % 2

15 % 3

number = 15

for num in range(2, number):
    if number % num == 0:
        print("Number {n1} is not prime".format(n1=number))
        break
# print("Number {} is prime".format(number))

list(range(1, 15))

number = 15
is_prime = True

for num in range(2, number):
    if number % num == 0:
        is_prime = False
        break

if is_prime:
    print("Number {} is prime".format(number))
else:
    print("Number {} is not prime".format(number))

number = 17
is_prime = True

for num in range(2, number):
    if number % num == 0:
        is_prime = False
        break

if is_prime:
    print("Number {} is prime".format(number))
else:
    print("Number {} is not prime".format(number))

def is_prime():
    _is_prime = True

    for num in range(2, number):
        if number % num == 0:
            _is_prime = False
            break
    print(_is_prime)

# function call
is_prime()

number = 23
is_prime()

number = 40
is_prime()

def is_prime(number):
    _is_prime = True

    for num in range(2, number):
        if number % num == 0:
            _is_prime = False
            break
    print(_is_prime)

is_prime()

is_prime(43)

is_prime(number=43)

if is_prime(43):
    print("This is prime number")

type(is_prime(43))

if None:
    print("True")
else:
    print("False")

def is_prime(number):
    _is_prime = True

    for num in range(2, number):
        if number % num == 0:
            _is_prime = False
            break
    return _is_prime

is_prime(41)

is_prime(42)

if is_prime(987):
    print("This is prime number")
else:
    print("not a prime number")

if is_prime(43):
    print("This is prime number")
else:
    print("not a prime number")

get_ipython().magic('timeit is_prime(1193)')

1193

import math

dir(math)

help(math.sqrt)

math.sqrt(1193)

int(math.sqrt(1193))

from math import sqrt

sqrt(1193)



from math import sqrt

def is_prime(number):
    _is_prime = True

    for num in range(2, int(sqrt(number)) + 1):
        if number % num == 0:
            _is_prime = False
            break
    return _is_prime

is_prime(2)

sqrt(2)

is_prime(3)

is_prime(4)

is_prime(23)

is_prime(1193)

get_ipython().magic('timeit is_prime(1193)')







