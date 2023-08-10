import sys 

sys.float_info

sys.float_info.max

infinity = float("inf")
infinity

infinity/10000

import math

print 'π: %.30f' % math.pi
print 'e: %.30f' % math.e

float("inf")-float("inf")

import math

print '{:^3}  {:6}  {:6}  {:6}'.format('e', 'x', 'x**2', 'isinf')
print '{:-^3}  {:-^6}  {:-^6}  {:-^6}'.format('', '', '', '')

for e in range(0, 201, 20):
    x = 10.0 ** e
    y = x*x
    print '{:3d}  {!s:6}  {!s:6}  {!s:6}'.format(e, x, y, math.isinf(y))

x = 10.0 ** 200

print 'x    =', x
print 'x*x  =', x*x
try:
    print 'x**2 =', x**2
except OverflowError, err:
    print err

import math

x = (10.0 ** 200) * (10.0 ** 200)
y = x/x

print 'x =', x
print 'isnan(x) =', math.isnan(x)
print 'y = x / x =', x/x
print 'y == nan =', y == float('nan')
print 'isnan(y) =', math.isnan(y)

import math

print '{:^5}  {:^5}  {:^5}  {:^5}  {:^5}'.format('i', 'int', 'trunk', 'floor', 'ceil')
print '{:-^5}  {:-^5}  {:-^5}  {:-^5}  {:-^5}'.format('', '', '', '', '')

fmt = '  '.join(['{:5.1f}'] * 5)

for i in [ -1.5, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 1 ]:
    print fmt.format(i, int(i), math.trunc(i), math.floor(i), math.ceil(i))

import math

for i in range(6):
    print '{}/2 = {}'.format(i, math.modf(i/2.0))

import math

print '{:^7}  {:^7}  {:^7}'.format('x', 'm', 'e')
print '{:-^7}  {:-^7}  {:-^7}'.format('', '', '')

for x in [ 0.1, 0.5, 4.0 ]:
    m, e = math.frexp(x)
    print '{:7.2f}  {:7.2f}  {:7d}'.format(x, m, e)

import math

print '{:^7}  {:^7}  {:^7}'.format('m', 'e', 'x')
print '{:-^7}  {:-^7}  {:-^7}'.format('', '', '')

for m, e in [ (0.8, -3),
              (0.5,  0),
              (0.5,  3),
              ]:
    x = math.ldexp(m, e)
    print '{:7.2f}  {:7d}  {:7.2f}'.format(m, e, x)

import math

print math.fabs(-1.1)
print math.fabs(-0.0)
print math.fabs(0.0)
print math.fabs(1.1)

import math

print
print '{:^5}  {:^5}  {:^5}  {:^5}  {:^5}'.format('f', 's', '< 0', '> 0', '= 0')
print '{:-^5}  {:-^5}  {:-^5}  {:-^5}  {:-^5}'.format('', '', '', '', '')

for f in [ -1.0,
            0.0,
            1.0,
            float('-inf'),
            float('inf'),
            float('-nan'),
            float('nan'),
            ]:
    s = int(math.copysign(1, f))
    print '{:5.1f}  {:5d}  {!s:5}  {!s:5}  {!s:5}'.format(f, s, f < 0, f > 0, f==0)

import math

values = [ 0.1 ] * 10

print 'Input values:', values

print 'sum()       : {:.20f}'.format(sum(values))

s = 0.0
for i in values:
    s += i
print 'for-loop    : {:.20f}'.format(s)
    
print 'math.fsum() : {:.20f}'.format(math.fsum(values))

import math

for i in [ 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.1 ]:
    try:
        print '{:2.0f}  {:6.0f}'.format(i, math.factorial(i))
    except ValueError, err:
        print 'Error computing factorial(%s):' % i, err

import math

print '{:^4}  {:^4}  {:^5}  {:^5}'.format('x', 'y', '%', 'fmod')
print '----  ----  -----  -----'

for x, y in [ (5, 2),
              (5, -2),
              (-5, 2),
              ]:
    print '{:4.1f}  {:4.1f}  {:5.2f}  {:5.2f}'.format(x, y, x % y, math.fmod(x, y))

import math

for x, y in [
    # Typical uses
    (2, 3),
    (2.1, 3.2),

    # Always 1
    (1.0, 5),
    (2.0, 0),

    # Not-a-number
    (2, float('nan')),

    # Roots
    (9.0, 0.5),
    (27.0, 1.0/3),
    ]:
    print '{:5.1f} ** {:5.3f} = {:6.3f}'.format(x, y, math.pow(x, y))

import math

print math.sqrt(9.0)
print math.sqrt(3)
try:
    print math.sqrt(-1)
except ValueError, err:
    print 'Cannot compute sqrt(-1):', err
    

import math

print '{:2}  {:^12}  {:^20}  {:^20}  {:8}'.format('i', 'x', 'accurate', 'inaccurate', 'mismatch')
print '{:-^2}  {:-^12}  {:-^20}  {:-^20}  {:-^8}'.format('', '', '', '', '')

for i in range(0, 10):
    x = math.pow(10, i)
    accurate = math.log10(x)
    inaccurate = math.log(x, 10)
    match = '' if int(inaccurate) == i else '*'
    print '{:2d}  {:12.1f}  {:20.18f}  {:20.18f}  {:^5}'.format(i, x, accurate, inaccurate, match)

import math

x = 2

fmt = '%.20f'
print fmt % (math.e ** 2)
print fmt % math.pow(math.e, 2)
print fmt % math.exp(2)

