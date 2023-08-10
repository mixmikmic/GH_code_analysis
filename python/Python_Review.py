a = 1
b = a
a+= 1
print(a)
print(b)

x = [1, 2, 3, 4, 5, ['a', 'b']]
y = x
x.append(100)
print("x is:   {0}".format(x))
print("y is:   {0}".format(y))
print("x is y: {0}".format(x is y))

z = x.copy()
x[5].append('c')
x.append(50)
print("x is:   {0}".format(x))
print("z is:   {0}".format(z))
print("x is z: {0}".format(x is z))

[0, 1, 2, 3, 4]

# run loop 5 times:
for i in range(5):
    print(i, end=" ")

for i in range(1, 70):
    if i%7:
        continue
    print(i, end=" ")

"Hello World!".upper()

"Hello World!".lower()

" Hello World! ".strip()

"10".isdigit()

"x={0}, y={1}".format(1, 2)

"x={x:3}, y={y:5.2f}".format(x=1, y=2)

import math
# loads math module and makes is usable in the current context (file)
# e.g.:
math.cos(math.pi)

from math import cos, pi
# only import cos() function and pi constant, but not sin(), etc.
# e.g.:
cos(2*pi)

import math as m
# import math module, but give it a different name, e.g. m
# e.g.:
m.cos(m.pi)

from math import cos as cosine
# only import cos() function and pi constant, but not sin(), etc.
# e.g.:
cosine(2*pi)

