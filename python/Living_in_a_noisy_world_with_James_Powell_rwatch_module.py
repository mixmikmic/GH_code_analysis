import numpy as np

np.random.seed(1234)
np.random.normal()

get_ipython().run_cell_magic('bash', '', 'tmpdir=$(mktemp -d)\ncd $tmpdir\ngit clone https://github.com/dutc/rwatch\ncd rwatch/src/\nmake\nls -larth ./rwatch.so\nfile ./rwatch.so\n# cp ./rwatch.so /where/ver/you/need/  # ~/publis/notebook/ for me')

import rwatch
from sys import setrwatch, getrwatch

setrwatch({})  # clean any previously installed rwatch
getrwatch()

from collections import defaultdict

def basic_view(frame, obj):
    print("Python saw the object {} from frame {}".format(obj, frame))
    return obj

x = "I am alive!"

setrwatch({
    id(x): basic_view
})

print(x)

def delrwatch(idobj):
    getrwatch().pop(idobj, None)

print(x)
delrwatch(id(x))
print(x)  # no more rwatch on this!
print(x)  # no more rwatch on this!

y = "I am Zorro !"
print(y)
delrwatch(y)  # No issue!
print(y)

from inspect import getframeinfo

def debug_view(frame, obj):
    info = getframeinfo(frame)
    msg = '- Access to {!r} (@{}) at {}:{}:{}'
    print(msg.format(obj, hex(id(obj)), info.filename, info.lineno, info.function))
    return obj

setrwatch({})
setrwatch({
    id(x): debug_view
})
getrwatch()

print(x)

setrwatch({})

def debug_view_for_str(frame, obj):
    if isinstance(obj, str):
        info = getframeinfo(frame)
        if '<stdin>' in info.filename or '<ipython-' in info.filename:
            msg = '- Access to {!r} (@{}) at {}:{}:{}'
            print(msg.format(obj, hex(id(obj)), info.filename, info.lineno, info.function))
    return obj

setrwatch(defaultdict(lambda: debug_view_for_str))

print(x)

setrwatch({})

def debug_view_for_any_object(frame, obj):
    info = getframeinfo(frame)
    if '<stdin>' in info.filename or '<ipython-' in info.filename:
        msg = '- Access to {!r} (@{}) at {}:{}:{}'
        print(msg.format(obj, hex(id(obj)), info.filename, info.lineno, info.function))
    return obj

print(x)
get_ipython().magic('time 123 + 134')

setrwatch({})
setrwatch(defaultdict(lambda: debug_view_for_any_object))
print(x)
get_ipython().magic('time 123 + 134')
setrwatch({})

class InspectThisObject(object):
    def __init__(self, obj):
        self.idobj = id(obj)
    
    def __enter__(self):
        getrwatch()[self.idobj] = debug_view

    def __exit__(self, exc_type, exc_val, exc_tb):
        delrwatch(self.idobj)

z = "I am Batman!"
print(z)

with InspectThisObject(z):
    print(z)

print(z)

class InspectAllObjects(object):
    def __init__(self):
        pass

    def __enter__(self):
        setrwatch(defaultdict(lambda: debug_view_for_any_object))

    def __exit__(self, exc_type, exc_val, exc_tb):
        setrwatch({})

with InspectAllObjects():
    print(0)

with InspectAllObjects():
    print("Darth Vader -- No Luke, I am your Father!")
    print("Luke -- I have a father? Yay! Let's eat cookies together!")

from numbers import Number

def add_white_noise_to_numbers(frame, obj):
    if isinstance(obj, Number):
        info = getframeinfo(frame)
        if '<stdin>' in info.filename or '<ipython-' in info.filename:
            return obj + np.random.normal()
    return obj

np.random.seed(1234)
setrwatch({})
x = 1234
print(x)
getrwatch()[id(x)] = add_white_noise_to_numbers
print(x)  # huhoww, that's noisy!
print(10 * x + x + x**2)  # and noise propagate!
setrwatch({})
print(x)
print(10 * x + x + x**2)

def add_white_noise_to_complex(frame, obj):
    if isinstance(obj, complex):
        info = getframeinfo(frame)
        if '<stdin>' in info.filename or '<ipython-' in info.filename:
            return obj + np.random.normal() + np.random.normal() * 1j
    return obj

np.random.seed(1234)
setrwatch({})
y = 1234j
print(y)
setrwatch(defaultdict(lambda: add_white_noise_to_complex))
print(y)  # huhoww, that's noisy!
setrwatch({})
print(y)

class WhiteNoiseComplex(object):
    def __init__(self):
        pass

    def __enter__(self):
        setrwatch(defaultdict(lambda: add_white_noise_to_complex))

    def __exit__(self, exc_type, exc_val, exc_tb):
        setrwatch({})

np.random.seed(120193)
print(120193, 120193j)
with WhiteNoiseComplex():
    print(120193, 120193j)  # Huhoo, noisy!
print(120193, 120193j)

print(0*1j)
with WhiteNoiseComplex():
    print(0*1j)  # Huhoo, noisy!
print(0*1j)

class Noisy(object):
    def __init__(self, noise):
        def add_white_noise_to_complex(frame, obj):
            if isinstance(obj, complex):
                info = getframeinfo(frame)
                if '<stdin>' in info.filename or '<ipython-' in info.filename:
                    return noise(obj)
            return obj

        self.rwatch = add_white_noise_to_complex

    def __enter__(self):
        setrwatch(defaultdict(lambda: self.rwatch))

    def __exit__(self, exc_type, exc_val, exc_tb):
        setrwatch({})

print(1j)
with Noisy(lambda obj: obj + np.random.normal()):
    print(1j)
print(1j)

print(1j)
with Noisy(lambda obj: obj * np.random.normal()):
    print(1j)
print(1j)

print(1j)
with Noisy(lambda obj: obj + np.random.normal(10, 0.1) + np.random.normal(10, 0.1) * 1j):
    print(1j)
print(1j)

