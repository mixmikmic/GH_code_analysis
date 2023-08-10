import ctypes

libc = ctypes.cdll.LoadLibrary("libc.so.6")

libc.time(None)

import time
libc.time(None), time.time()

libc.printf("Hello from C!\n")

libc.printf("one %d two %g three %s\n",
            ctypes.c_int(1),
            ctypes.c_double(2.2),
            ctypes.c_char_p("THREE"))

gslcblas = ctypes.CDLL("libgslcblas.so", mode=ctypes.RTLD_GLOBAL)
gsl = ctypes.CDLL("libgsl.so")

# my function
def f(x, params):
    out = 3.0*x**2 + 2.0*x + 1.0
    print "f({0}) = {1}".format(x, out)
    return out

# callback with the appropriate signature
func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.POINTER(None))

# wrapped up in the struct that GSL expects
class gsl_function(ctypes.Structure):
    _fields_ = [("f", func_type),
                ("params", ctypes.POINTER(None))]

# finally, make the object
callback = gsl_function(func_type(f), 0)

result = ctypes.c_double(-1.0);
abserr = ctypes.c_double(-1.0);

p_result = ctypes.POINTER(ctypes.c_double)(result)
p_abserr = ctypes.POINTER(ctypes.c_double)(abserr)

gsl.gsl_deriv_central(ctypes.POINTER(gsl_function)(callback), ctypes.c_double(2.0), ctypes.c_double(1e-8), p_result, p_abserr)

p_result[0], p_abserr[0]

if hasattr(ctypes.pythonapi, "Py_InitModule4_64"):
    Py_ssize_t = ctypes.c_int64   # modern versions of Python 2 use 64-bit ints almost everywhere
else:
    Py_ssize_t = ctypes.c_int     # very old versions of Python used plain C ints

class PyObject(ctypes.Structure): pass
PyObject._fields_ = [("ob_refcnt", Py_ssize_t),
                     ("ob_type", ctypes.POINTER(PyObject))]

class SlotsPointer(PyObject):
    _fields_ = [("dict", ctypes.POINTER(PyObject))]

def proxy_builtin(cls):
    name = cls.__name__
    slots = getattr(cls, "__dict__", name)

    pointer = SlotsPointer.from_address(id(slots))
    namespace = {}

    ctypes.pythonapi.PyDict_SetItem(
        ctypes.py_object(namespace),
        ctypes.py_object(name),
        pointer.dict
    )

    return namespace[name]

# don't call this twice (you'll crash the interpreter)

def do_crazy_things(lst):
    print "crazy {0}".format(lst)

proxy_builtin(list)["crazy"] = property(do_crazy_things)
hasattr([], "crazy")

[1, 2, 3, 4, 5].crazy

