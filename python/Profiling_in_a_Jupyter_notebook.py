get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -v -m -g')

def recip_square(i):
    return 1./i**2

def approx_pi(n=10000000):
    val = 0.
    for k in range(1,n+1):
        val += recip_square(k)
    return (6 * val)**.5

get_ipython().magic('time approx_pi()')

get_ipython().magic('timeit approx_pi()')

# Thanks to https://nbviewer.jupyter.org/gist/minrk/7715212
from __future__ import print_function
from IPython.core import page
def myprint(s):
    try:
        print(s['text/plain'])
    except (KeyError, TypeError):
        print(s)
page.page = myprint

get_ipython().magic('prun approx_pi()')

get_ipython().magic('lprun -T /tmp/test_lprun.txt -f recip_square -f approx_pi approx_pi()')
get_ipython().magic('cat /tmp/test_lprun.txt')

get_ipython().magic('load_ext memory_profiler')

get_ipython().magic('mprun -T /tmp/test_mprun.txt -f recip_square -f approx_pi approx_pi()')
get_ipython().magic('cat /tmp/test_mprun.txt')

get_ipython().magic('memit approx_pi()')

get_ipython().magic('load_ext cython')

get_ipython().run_cell_magic('cython', '-3', '# cython: profile=True\n\ndef recip_square2(int i):\n    return 1./i**2\n\ndef approx_pi2(int n=10000000):\n    cdef double val = 0.\n    cdef int k\n    for k in range(1, n + 1):\n        val += recip_square2(k)\n    return (6 * val)**.5')

get_ipython().magic('timeit approx_pi()')
get_ipython().magic('timeit approx_pi2()')

get_ipython().magic('prun approx_pi2()')

get_ipython().run_cell_magic('cython', '-3', '# cython: profile=True\n\ncdef double recip_square3(int i):\n    return 1./(i**2)\n\ndef approx_pi3(int n=10000000):\n    cdef double val = 0.\n    cdef int k\n    for k in range(1, n + 1):\n        val += recip_square3(k)\n    return (6 * val)**.5')

get_ipython().magic('timeit approx_pi3()')

get_ipython().magic('prun approx_pi3()')

get_ipython().run_cell_magic('cython', '-3', '\ncdef inline double recip_square4(int i):\n    return 1./(i**2)\n\ndef approx_pi4(int n=10000000):\n    cdef double val = 0.\n    cdef int k\n    for k in range(1, n + 1):\n        val += recip_square4(k)\n    return (6 * val)**.5')

get_ipython().magic('timeit approx_pi4()')

get_ipython().magic('prun approx_pi4()')

get_ipython().run_cell_magic('cython', '-3', '# cython: profile=True\nfrom __future__ import division\nimport cython\n\n@cython.profile(False)\ncdef inline double recip_square5(int i):\n    return 1./(i*i)\n\ndef approx_pi5(int n=10000000):\n    cdef double val = 0.\n    cdef int k\n    for k in range(1, n + 1):\n        val += recip_square5(k)\n    return (6 * val)**.5')

get_ipython().magic('timeit approx_pi4()')

get_ipython().magic('prun approx_pi4()')

