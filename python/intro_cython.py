get_ipython().magic('load_ext cython')

get_ipython().run_cell_magic('cython', '-a', 'def py_fib(n):\n    a, b = 1, 1\n    for i in range(n):\n        a, b = a + b, a\n    return a\n\ndef cy_fib(int n):\n    cdef int i, a, b\n    a, b = 1, 1\n    for i in range(n):\n        a, b = a + b, a\n    return a')

get_ipython().magic('timeit -n 10000 -r 3 py_fib(1000)')
get_ipython().magic('timeit -n 10000 -r 3 cy_fib(1000)')

def py_primes(kmax):
    """Calculation of prime numbers in standard Python syntax."""

    p = [0] * 1000
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result

print(py_primes(10))

get_ipython().run_cell_magic('cython', '-a', 'def cy_primes(int kmax):\n    """Calculation of prime numbers in Cython."""\n    cdef int n, k, i\n    cdef int p[1000]\n    result = []\n    if kmax > 1000:\n        kmax = 10000\n    k = 0\n    n = 2\n    while k < kmax:\n        i = 0\n        while i < k and n % p[i] != 0:\n            i = i + 1\n        if i == k:\n            p[k] = n\n            k = k + 1\n            result.append(n)\n        n = n + 1\n    return result\n\nprint(cy_primes(10))')

get_ipython().magic('timeit -n 10000 -r 3 py_primes(20)')
get_ipython().magic('timeit -n 10000 -r 3 cy_primes(20)')

# The Levenshtein distance between two words is the minimum number of single-character edits
# (insertions, deletions or substitutions) required to change one word into the other.

def py_levenshtein(s, t):
    return py_lev(s, len(s), t, len(t))

def py_lev(s, len_s, t, len_t):
    if len_s == 0 or len_t == 0:
        return len_s or len_t
    return min(py_lev(s, len_s-1, t, len_t) + 1,
               py_lev(s, len_s, t, len_t-1) + 1,
               py_lev(s, len_s-1, t, len_t-1) + py_cost(s, len_s, t, len_t)) 

def py_cost(s, len_s, t, len_t):
    return s[len_s-1] != t[len_t-1]

get_ipython().run_cell_magic('cython', '-a', 'def cy_levenshtein(s, t):\n    return lev(s, len(s), t, len(t))\n\ncdef int lev(char *s, int len_s, char *t, int len_t):\n    if len_s == 0 or len_t == 0:\n        return len_s or len_t\n    cdef:\n        int lev_s = lev(s, len_s-1, t, len_t  ) + 1\n        int lev_t = lev(s, len_s, t, len_t-1) + 1\n        int lev_b = lev(s, len_s-1, t, len_t-1) + cost(s, len_s, t, len_t)\n    if lev_s < lev_t and lev_s < lev_b:\n        return lev_s\n    elif lev_t < lev_s and lev_t < lev_b:\n        return lev_t\n    else:\n        return lev_b\n\ncdef int cost(char *s, int len_s, char *t, int len_t):\n    return s[len_s-1] != t[len_t-1]')

a = b'abcdefgh'
b = b'adqdfyhi'
get_ipython().magic('timeit -n 50 -r 3 py_levenshtein(a, b)')
get_ipython().magic('timeit -n 50 -r 3 cy_levenshtein(a, b)')

