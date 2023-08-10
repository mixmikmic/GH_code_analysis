import itertools

import numpy
from sympy import init_session

init_session()

C = symbols("c0:4")
S = symbols("s0:4")
A = symbols("a0:4")
B = symbols("b0:4")

def dot(es1, es2):
    return sum([e[0]*e[1] for e in zip(es1, es2)])

def do_integral(f, s):
    return Integral(f, (s, 0, 1))
        # The bounds of 0 and 1 are specific to the Fourier FA. Make
        # sure you normalize the inputs.

da_integral = reduce(do_integral, S[1:], cos(pi * dot(S, C))).doit()

# Copied from hiora_cartpole.fourier_fa.
order = 3
n_dims = 4
c_matrix = np.array(
               list( itertools.product(range(order+1), repeat=n_dims) ),
               dtype=np.int32)

def sum_term(integral, c, c_vec):
    return integral.subs(zip(c, c_vec))

sum_terms = [sum_term(da_integral, C, c_vec) for c_vec in c_matrix]

np_sum_terms = [lambdify(S[0], sum_term, 'numpy') for sum_term in sum_terms]

def phi(npsts, theta, s0):
    ns0 = (s0 - -2.5) / 5.0
    return np.dot(theta, 
                  np.array([npst(ns0) for npst in npsts]))

theta = np.load("theta.npy")

res = np.array([phi(np_sum_terms, theta[512:768], x) 
 for x in np.arange(-2.38, 2.5, 0.5*1.19)])

res

other = np.array([-3748598.374407076,
 -8333255.9176837215,
 92906846.75614552,
 242969379.49722022,
 320543060.70953935,
 257463642.38676718,
 107526913.72252564,
 -7061727.3744605975,
 -14631018.954087665])

res/other

other/res

res2 = np.array([phi(np_sum_terms, theta[768:1024], x) 
 for x in np.arange(-2.38, 2.5, 0.5*1.19)])

res2

res2*29.5



