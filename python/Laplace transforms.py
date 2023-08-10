import sympy
sympy.init_printing()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

t = sympy.Symbol('t', real=True)
s = sympy.Symbol('s')
a = sympy.Symbol('a', real=True, positive=True)

f = sympy.exp(-a*t)
f

sympy.laplace_transform(f, t, s)

F = sympy.laplace_transform(f, t, s, noconds=True)
F

def L(f):
    return sympy.laplace_transform(f, t, s, noconds=True)

def invL(F):
    return sympy.inverse_laplace_transform(F, s, t)

invL(F)

sympy.Heaviside(t)

sympy.plot(sympy.Heaviside(t));

invL(F).subs({a: 2})

p = sympy.plot(f.subs({a: 2}), invL(F).subs({a: 2}), 
               xlim=(-1, 4), ylim=(0, 3), show=False)
p[1].line_color = 'red'
p.show()

omega = sympy.Symbol('omega', real=True)
exp = sympy.exp
sin = sympy.sin
cos = sympy.cos
functions = [1,
         t,
         exp(-a*t),
         t*exp(-a*t),
         t**2*exp(-a*t),
         sin(omega*t),
         cos(omega*t),
         1 - exp(-a*t),
         exp(-a*t)*sin(omega*t),
         exp(-a*t)*cos(omega*t),
         ]
functions

Fs = [L(f) for f in functions]
Fs

from pandas import DataFrame

def makelatex(args):
    return ["${}$".format(sympy.latex(a)) for a in args]

DataFrame(list(zip(makelatex(functions), makelatex(Fs))))

F = ((s + 1)*(s + 2)* (s + 3))/((s + 4)*(s + 5)*(s + 6))

F

F.apart(s)

sympy.inverse_laplace_transform(F.apart(s), s, t)



