import sympy
sympy.init_printing()

s = sympy.Symbol('s')

def fopdt(k, theta, tau):
    return k*sympy.exp(-theta*s)/(tau*s + 1)

G_p = sympy.Matrix([[fopdt(-2, 1, 10), fopdt(1.5, 1, 1)],
                    [fopdt(1.5, 1, 1), fopdt(2, 1, 10)]])
G_p

#sympy.limit(G_p, s, 0)

def gain(G):
    return sympy.limit(G, s, 0)

K = G_p.applyfunc(gain)
K

Lambda = K.multiply_elementwise(K.inv().transpose())
Lambda

import numpy

def fopdt(k, theta, tau):
    return k*numpy.exp(-theta*s)/(tau*s + 1)

s = 0

K = numpy.matrix([[fopdt(-2, 1, 10), fopdt(1.5, 1, 1)],
                  [fopdt(1.5, 1, 1), fopdt(2, 1, 10)]])

K.A*K.I.T.A

