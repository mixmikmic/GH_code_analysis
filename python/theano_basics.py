from theano import *
import theano.tensor as T
import numpy as np #np.arrays are used for input and output

#Define a computational graph
x = T.dscalar()
y = T.dscalar()
z = x + y
f = function([x,y], z) #Compile in C

print(f)
print(z)
print(f(1, 2))
print(z.eval({x: 1, y: 2})) #eval does not compile, but it's not very flexible

A = T.dmatrix('A')
x = T.dvector('x')
b = T.dvector('b')
z = theano.dot(A, x) + b 
f = function([A, x, b], z)

A = np.array([[1, -1], [-1, 1]])
x = np.array([2, 1])
b = np.array([1, 1])
f(A, x, b)

from theano import shared
count = shared(0) #Creates a shareable "0"
inc = T.iscalar()
result = count + inc
acc = function([inc], result)

print(acc(1))
print(count.get_value())

count.set_value(acc(1))
print(count.get_value())
count.set_value(acc(1))
print(count.get_value())
count.set_value(acc(10))
print(count.get_value())

count = shared(0)
inc = T.iscalar()
acc = function([inc], count + inc, updates=[(count, count + inc)])

print(acc(1))
print(acc(1))
print(acc(20))
print(count.get_value())

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=0) 
num = srng.uniform()
f = function([], num)
g = function([], num, no_default_updates=True)

print("Update generator")
for _ in range(3):
    print(f())
print("\nDo not update gerador")
for _ in range(3):
    print(g())

from theano import pp #pretty print
x = T.dscalar('x')
y = x ** 2 
dy = T.grad(y, x) #dy/dx
f = theano.function([x], dy)
pp(dy)

print(f(3), f(0), f(1))



