def lempel_ziv_complexity(binary_sequence):
    """Lempel-Ziv complexity for a binary sequence, in simple Python code."""
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
    return complexity

s = '1001111011000010'
lempel_ziv_complexity(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010

get_ipython().magic('timeit lempel_ziv_complexity(s)')

lempel_ziv_complexity('1010101010101010')  # 1 / 0 / 10

lempel_ziv_complexity('1001111011000010000010')  # 1 / 0 / 01 / 1110

lempel_ziv_complexity('100111101100001000001010')  # 1 / 0 / 01 / 1110 / 1100 / 0010 / 000 / 010 / 10

get_ipython().magic("timeit lempel_ziv_complexity('100111101100001000001010')")

get_ipython().magic('load_ext cython')

get_ipython().run_cell_magic('cython', '', 'from __future__ import division\nimport cython\n\nctypedef unsigned int DTYPE_t\n\n@cython.boundscheck(False) # turn off bounds-checking for entire function, quicker but less safe\ndef lempel_ziv_complexity_cython(str binary_sequence not None):\n    """Lempel-Ziv complexity for a binary sequence, in simple Cython code (C extension)."""\n    cdef DTYPE_t u = 0\n    cdef DTYPE_t v = 1\n    cdef DTYPE_t w = 1\n    cdef DTYPE_t v_max = 1\n    cdef DTYPE_t length = len(binary_sequence)\n    cdef DTYPE_t complexity = 1\n    # that was the only needed part, typing statically all the variables\n    while True:\n        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:\n            v += 1\n            if w + v >= length:\n                complexity += 1\n                break\n        else:\n            if v > v_max:\n                v_max = v\n            u += 1\n            if u == w:\n                complexity += 1\n                w += v_max\n                if w > length:\n                    break\n                else:\n                    u = 0\n                    v = 1\n                    v_max = 1\n            else:\n                v = 1\n    return complexity')

s = '1001111011000010'
lempel_ziv_complexity_cython(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010

get_ipython().magic('timeit lempel_ziv_complexity_cython(s)')

lempel_ziv_complexity_cython('1010101010101010')  # 1 / 0 / 10

lempel_ziv_complexity_cython('1001111011000010000010')  # 1 / 0 / 01 / 1110

lempel_ziv_complexity_cython('100111101100001000001010')  # 1 / 0 / 01 / 1110 / 1100 / 0010 / 000 / 010 / 10

get_ipython().magic("timeit lempel_ziv_complexity_cython('100111101100001000001010')")

from numba import jit

@jit("int32(boolean[:])")
def lempel_ziv_complexity_numba_x(binary_sequence):
    """Lempel-Ziv complexity for a binary sequence, in Python code using numba.jit() for automatic speedup (hopefully)."""
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
    return complexity

def str_to_numpy(s):
    """str to np.array of bool"""
    return np.array([int(i) for i in s], dtype=np.bool)

def lempel_ziv_complexity_numba(s):
    return lempel_ziv_complexity_numba_x(str_to_numpy(s))

str_to_numpy(s)

s = '1001111011000010'
lempel_ziv_complexity_numba(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010

get_ipython().magic('timeit lempel_ziv_complexity_numba(s)')

lempel_ziv_complexity_numba('1010101010101010')  # 1 / 0 / 10

lempel_ziv_complexity_numba('1001111011000010000010')  # 1 / 0 / 01 / 1110

lempel_ziv_complexity_numba('100111101100001000001010')  # 1 / 0 / 01 / 1110 / 1100 / 0010 / 000 / 010 / 10

get_ipython().magic("timeit lempel_ziv_complexity_numba('100111101100001000001010')")

from numpy.random import binomial

def bernoulli(p, size=1):
    """One or more samples from a Bernoulli of probability p."""
    return binomial(1, p, size)

bernoulli(0.5, 20)

''.join(str(i) for i in bernoulli(0.5, 20))

def random_binary_sequence(n, p=0.5):
    """Uniform random binary sequence of size n, with rate of 0/1 being p."""
    return ''.join(str(i) for i in bernoulli(p, n))

random_binary_sequence(50)
random_binary_sequence(50, p=0.1)
random_binary_sequence(50, p=0.25)
random_binary_sequence(50, p=0.5)
random_binary_sequence(50, p=0.75)
random_binary_sequence(50, p=0.9)

def tests_3_functions(n, p=0.5, debug=True):
    s = random_binary_sequence(n, p=p)
    c1 = lempel_ziv_complexity(s)
    if debug:
        print("Sequence s = {} ==> complexity C = {}".format(s, c1))
    c2 = lempel_ziv_complexity_cython(s)
    c3 = lempel_ziv_complexity_numba(s)
    assert c1 == c2 == c3, "Error: the sequence {} gave different values of the Lempel-Ziv complexity from 3 functions ({}, {}, {})...".format(s, c1, c2, c3)
    return c1

tests_3_functions(5)

tests_3_functions(20)

tests_3_functions(50)

tests_3_functions(500)

tests_3_functions(5000)

get_ipython().magic("timeit lempel_ziv_complexity('100111101100001000001010')")
get_ipython().magic("timeit lempel_ziv_complexity_cython('100111101100001000001010')")
get_ipython().magic("timeit lempel_ziv_complexity_numba('100111101100001000001010')")

get_ipython().magic("timeit lempel_ziv_complexity('10011110110000100000101000100100101010010111111011001111111110101001010110101010')")
get_ipython().magic("timeit lempel_ziv_complexity_cython('10011110110000100000101000100100101010010111111011001111111110101001010110101010')")
get_ipython().magic("timeit lempel_ziv_complexity_numba('10011110110000100000101000100100101010010111111011001111111110101001010110101010')")

get_ipython().magic('timeit tests_3_functions(10, debug=False)')
get_ipython().magic('timeit tests_3_functions(20, debug=False)')
get_ipython().magic('timeit tests_3_functions(40, debug=False)')
get_ipython().magic('timeit tests_3_functions(80, debug=False)')
get_ipython().magic('timeit tests_3_functions(160, debug=False)')
get_ipython().magic('timeit tests_3_functions(320, debug=False)')

def test_cython(n):
    s = random_binary_sequence(n)
    c = lempel_ziv_complexity_cython(s)
    return c

get_ipython().magic('timeit test_cython(10)')
get_ipython().magic('timeit test_cython(20)')
get_ipython().magic('timeit test_cython(40)')
get_ipython().magic('timeit test_cython(80)')
get_ipython().magic('timeit test_cython(160)')
get_ipython().magic('timeit test_cython(320)')

get_ipython().magic('timeit test_cython(640)')
get_ipython().magic('timeit test_cython(1280)')
get_ipython().magic('timeit test_cython(2560)')
get_ipython().magic('timeit test_cython(5120)')

get_ipython().magic('timeit test_cython(10240)')
get_ipython().magic('timeit test_cython(20480)')

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(context="notebook", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.4)

x = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480]
y = [18, 30, 55, 107, 205, 471, 977, 2270, 5970, 17300, 56600, 185000]

plt.figure()
plt.plot(x, y, 'o-')
plt.xlabel("Length $n$ of the binary sequence $S$")
plt.ylabel(r"Time in $\mu\;\mathrm{s}$")
plt.title("Time complexity of Lempel-Ziv complexity")
plt.show()

plt.figure()
plt.semilogx(x, y, 'o-')
plt.xlabel("Length $n$ of the binary sequence $S$")
plt.ylabel(r"Time in $\mu\;\mathrm{s}$")
plt.title("Time complexity of Lempel-Ziv complexity, semilogx scale")
plt.show()

plt.figure()
plt.semilogy(x, y, 'o-')
plt.xlabel("Length $n$ of the binary sequence $S$")
plt.ylabel(r"Time in $\mu\;\mathrm{s}$")
plt.title("Time complexity of Lempel-Ziv complexity, semilogy scale")
plt.show()

plt.figure()
plt.loglog(x, y, 'o-')
plt.xlabel("Length $n$ of the binary sequence $S$")
plt.ylabel(r"Time in $\mu\;\mathrm{s}$")
plt.title("Time complexity of Lempel-Ziv complexity, loglog scale")
plt.show()

get_ipython().run_cell_magic('time', '', '%%script julia\n\n"""Lempel-Ziv complexity for a binary sequence, in simple Julia code."""\nfunction lempel_ziv_complexity(binary_sequence)\n    u, v, w = 0, 1, 1\n    v_max = 1\n    size = length(binary_sequence)\n    complexity = 1\n    while true\n        if binary_sequence[u + v] == binary_sequence[w + v]\n            v += 1\n            if w + v >= size\n                complexity += 1\n                break\n            end\n        else\n            if v > v_max\n                v_max = v\n            end\n            u += 1\n            if u == w\n                complexity += 1\n                w += v_max\n                if w > size\n                    break\n                else\n                    u = 0\n                    v = 1\n                    v_max = 1\n                end\n            else\n                v = 1\n            end\n        end\n    end\n    return complexity\nend\n\ns = "1001111011000010"\nlempel_ziv_complexity(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010\n\nM = 100;\nN = 10000;\nfor _ in 1:M\n    s = join(rand(0:1, N));\n    lempel_ziv_complexity(s);\nend\nlempel_ziv_complexity(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010')

get_ipython().run_cell_magic('time', '', '%%pypy\n\ndef lempel_ziv_complexity(binary_sequence):\n    """Lempel-Ziv complexity for a binary sequence, in simple Python code."""\n    u, v, w = 0, 1, 1\n    v_max = 1\n    length = len(binary_sequence)\n    complexity = 1\n    while True:\n        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:\n            v += 1\n            if w + v >= length:\n                complexity += 1\n                break\n        else:\n            if v > v_max:\n                v_max = v\n            u += 1\n            if u == w:\n                complexity += 1\n                w += v_max\n                if w > length:\n                    break\n                else:\n                    u = 0\n                    v = 1\n                    v_max = 1\n            else:\n                v = 1\n    return complexity\n\ns = "1001111011000010"\nlempel_ziv_complexity(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010\n\nfrom random import random\n\nM = 100\nN = 10000\nfor _ in range(M):\n    s = \'\'.join(str(int(random() < 0.5)) for _ in range(N))\n    lempel_ziv_complexity(s)')

