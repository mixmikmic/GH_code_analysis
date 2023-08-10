powers = (lambda x: pow(x, n) for n in range(-4,5))
phi = (1 + pow(5,0.5)) * 0.5 # golden proportion
for n, f in enumerate(powers, start=-4):  # iterates through lambda expressions
    print("phi ** {:2} == {:10.8f}".format(n, f(phi)))

class Any:
    
    def __init__(self):
        self.__dict__ = {0:'scissors', 1:'paper', 2:'rock'}
    
    def __getitem__(self, n):  # enough for iter() to go on
        if n == len(self.__dict__):
            raise StopIteration  # tells for loop when to stop
        return self.__dict__[n]
    
for thing in Any():
    print(thing)

import pprint

def primes():
    """generate successive prime numbers (trial by division)"""
    candidate = 1
    _primes_so_far = [2]     # first prime, only even prime
    yield _primes_so_far[0]  # share it!
    while True:
        candidate += 2    # check odds only from now on
        for prev in _primes_so_far:
            if prev**2 > candidate:
                yield candidate  # new prime!
                _primes_so_far.append(candidate)
                break
            if not divmod(candidate, prev)[1]: # no remainder!
                break                          # done looping
                
p = primes()  # generator function based iterator
pp = pprint.PrettyPrinter(width=40, compact=True)
pp.pprint([next(p) for _ in range(30)])  # next 30 primes please!

class Primes:

    def __init__(self):
        self.candidate = 1
        self._primes_so_far = [2]  # first prime, only even prime

    def __iter__(self):
        return self
        
    def __next__(self):
        while True:
            self.candidate += 2    # check odds only from now on
            for prev in self._primes_so_far:
                if prev**2 > self.candidate:
                    self._primes_so_far.append(self.candidate)
                    return self._primes_so_far[-2]    
                if not divmod(self.candidate, prev)[1]: # no remainder!
                    break

pp = pprint.PrettyPrinter(width=40, compact=True)
p = Primes()  # class based iterator
pp.pprint([next(p) for _ in range(30)])  # n

from itertools import islice
p = Primes()
for n in islice(p, 0, 20):
    print(n, end=", ")

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:48:52 2016

@author: Kirby Urner

David Beazley:
https://youtu.be/Z_OAlIhXziw?t=23m42s

Trial by division, but this time the primes coroutine acts 
more as a filter, passing qualified candidates through to
print_me, which writes to a file.
"""
import pprint

def coroutine(func):
    """
    Advances decorated generator function to the first yield
    """
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        cr.send(None)  # or next(cr) or cr.__next__()
        return cr
    return start
        
@coroutine
def print_me(file_name):
    with open(file_name, 'w') as file_obj:
        while True:
            to_print = (yield)
            file_obj.write(str(to_print)+"\n")
    
@coroutine
def primes(target):
    _primes_so_far = [2]
    target.send(2)
    while True:
        candidate = (yield)
        for prev in _primes_so_far:
            if not divmod(candidate, prev)[1]:
                break
            if prev**2 > candidate:
                _primes_so_far.append(candidate)
                target.send(candidate)
                break

output = print_me("primes.txt")
p = primes(output)

for x in range(3, 200, 2):  # test odds 3-199
    p.send(x)

with open("primes.txt", 'r') as file_obj:
    print(", ".join(file_obj.read().split("\n"))[:-2])

