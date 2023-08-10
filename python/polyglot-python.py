files = get_ipython().getoutput('ls')
files

get_ipython().run_cell_magic('bash', '', 'uname -a\necho $PWD')

# requires `conda install cython`
get_ipython().magic('load_ext Cython')

def f(x):
    return x**2-x

def integrate_f(a, b, N):
    s = 0; dx = (b-a)/N
    for i in range(N):
        s += f(a+i*dx)
    return s * dx

get_ipython().run_cell_magic('cython', '', 'cdef double fcy(double x) except? -2:\n    return x**2-x\n\ndef integrate_fcy(double a, double b, int N):\n    cdef int i\n    cdef double s, dx\n    s = 0; dx = (b-a)/N\n    for i in range(N):\n        s += fcy(a+i*dx)\n    return s * dx')

get_ipython().magic('timeit integrate_f(0, 1, 100)')
get_ipython().magic('timeit integrate_fcy(0, 1, 100)')

get_ipython().magic('pinfo %%cython')

get_ipython().run_cell_magic('cython', '-lm', "# Link the m library (like g++ linker argument)\nfrom libc.math cimport sin\nprint 'sin(1)=', sin(1)")

get_ipython().run_cell_magic('cython', '-a', 'cdef double fcy(double x) except? -2:\n    return x**2-x\n\ndef integrate_fcy(double a, double b, int N):\n    cdef int i\n    cdef double s, dx\n    s = 0; dx = (b-a)/N\n    for i in range(N):\n        s += fcy(a+i*dx)\n    return s * dx')

