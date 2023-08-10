import sympy

sympy.init_printing()

x = sympy.Symbol('x', positive=True)
s = sympy.Symbol('s', positive=True)

cauchy = 2 / (sympy.pi * (1 + x ** 2))
normal = 2 / sympy.sqrt(2*sympy.pi) * sympy.exp(-x**2 / 2)

hs = sympy.integrate(normal * cauchy.subs({x: s / x}) / x, (x, 0, sympy.oo))
hs

beta = sympy.Symbol('beta', positive=True)
gamma = sympy.Symbol('gamma', positive=True)
theta = sympy.Symbol('theta')
z1 = sympy.integrate(sympy.cos(theta)**(2*((gamma/(beta - 2)) - 3/2) + 3),
                    (theta, -sympy.pi, sympy.pi))
z1

beta = sympy.Symbol('beta', positive=True)
gamma = sympy.Symbol('gamma', constant = True)
theta = sympy.Symbol('theta')
z1 = sympy.integrate(sympy.cos(theta)**(2*(gamma/(beta - 2))),
                    (theta, -sympy.pi/2, sympy.pi/2))
z1

from sympy.stats import Beta, Binomial, density, E, variance
from sympy import Symbol, simplify, pprint, expand_func

alpha = Symbol("alpha", positive=True)
beta = Symbol("beta", positive=True)
z = Symbol("z")
theta = Beta("theta", alpha, beta)
D = density(theta)(z)
pprint(D, use_unicode=False)
expand_func(simplify(E(theta, meijerg=True)))



normal = 1 / sympy.sqrt(2*sympy.pi) * sympy.exp(-x**2 / 2)

x = sympy.Symbol('x')
normal = 1 / sympy.sqrt(2*sympy.pi) * sympy.exp(-x**2 / 2)
sympy.integrate(normal, (x, -sympy.oo, sympy.oo))

n = sympy.Symbol('n', integer=True, positive=True)
k = sympy.Symbol('k', integer=True, positive=True)
a = b = 1.
p = .25
pr = sympy.binomial(n, k)*(p**k)*(1-p)**(n-k)
pr

sympy.summation(pr*k, (k, 0, 20), (n, 20, 20))

a = .5
b = 1.
p = x**(a-1)*(1-x)**(b-1)/sympy.beta(a, b)
n = 10
k = sympy.Symbol('k', positive=True)
pr = sympy.binomial(n, k)*(p**k)*(1-p)**(n-k)
sympy.integrate(pr.subs({x: k}), (x, 0, n))

a = 1.
b = 3.
n = 20
k = sympy.Symbol('k', integer=True, positive=True)
p = sympy.binomial(n, k)*sympy.beta(k+a, n-k+b)/sympy.beta(a, b)
sympy.summation(p*k, (k, 0, 20))

x = sympy.Symbol('x', positive=True)
a = 1.#sympy.Symbol('alpha', positive=True)
b = 1.#sympy.Symbol('beta', positive=True)
beta = (x**(a-1))*((1-x)**(b-1))/sympy.beta(a, b)
sympy.integrate(beta*x, (x, 0, 1))

sympy.beta(1, 1)

