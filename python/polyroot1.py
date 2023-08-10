from sympy import N, roots
from sympy import Integer as Int

r= roots([1,-21,120,-100])
for x in r:
    print x

r= roots([Int(99)/100,-21,120,-100])
for x in r:
    print N(x)

r= roots([Int(101)/100,-21,120,-100])
for x in r:
    print N(x)

