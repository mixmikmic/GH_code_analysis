import numba

def pyfunc(x):
    return 3.0*x**2 + 2.0*x + 1.0

pyfunc(3.14)

# a compiled function that has no Python objects in the body,
# but it can be used in Python because it interprets Python on the way in
@numba.jit(nopython=True)
def fastfunc(x):
    return 3.0*x**2 + 2.0*x + 1.0

fastfunc(3.14)

# have to provide a signature: "double (double, void*)"
sig = numba.types.double(numba.types.double,
                         numba.types.CPointer(numba.types.void))
# a pure C function that doesn't interpret Python arguments
@numba.cfunc(sig, nopython=True)
def cfunc(x, params):
    return 3.0*x**2 + 2.0*x + 1.0

cfunc(3.14)   # should raise an error

# we get a function pointer instead
cfunc.address

# just to verify that this pointer works, get ctypes to use it
import ctypes

func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.POINTER(None))

func_type(cfunc.address)(3.14, None)

