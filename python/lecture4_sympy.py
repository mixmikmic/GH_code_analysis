# import numpy, sympy, configure printing and plotting
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import seaborn as sns

# notebook config
sns.set()                                   # nice plotting defaults
sym.init_printing(use_latex='mathjax')      # render latex for output
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # vectorized plots instead of png")

# globals for whole notebook: symbol x, domain Omega
x = sym.Symbol('x')
L = 1
Omega = (0, L)

# convenience functions for derivation and integration
def ddx(f):
    """ derivative of sympy expr f w/r/t x """
    return sym.diff(f, x)

def integrate_domain(f):
    """ integrate sympy expression f over domain \Omega w/r/t x """
    return sym.integrate(f, (x, Omega[0], Omega[1]))

c1, c2, mu = sym.symbols('c1 c2 mu')

# general solution
gen = c1*sym.exp(sym.sqrt(mu)*x) + c2 * sym.exp(-sym.sqrt(mu)*x)

# check that it satisfies the ODE
assert sym.simplify(-ddx(ddx(gen)) + mu*gen) == 0
gen

# solve algebraic system for c1, c2 by applying boundary conditions
system = list()
system.append(gen.subs(x,0) - 1)   # solution @ x=0 is 1
system.append(ddx(gen).subs(x, L)) # derivative @ x=L is 0

from sympy.solvers.solveset import linsolve
C_solution = linsolve(system, (c1, c2))
C1, C2 = next(iter(C_solution))
print('C1: {}'.format(C1))
print('C2: {}'.format(C2))
C1, C2

# evaluate the exact solution using the solved-for coefficients
exact_solution = sym.simplify(gen.subs([(c1, C1), (c2, C2)]))
print('exact solution:\n{}'.format(exact_solution))
exact_solution

# check that exact solution satisfies the ODE, BCs
assert -ddx(ddx(exact_solution)) + mu * exact_solution == 0
assert ddx(exact_solution).subs(x, L) == 0
assert exact_solution.subs(x, 0) == 1

# render the latex of the solution to paste into *.tex file
ex_sol_latex = sym.latex(exact_solution)
print(ex_sol_latex)

# build a vectorized function to evaluate our exact solution
u_exact = sym.lambdify((x, mu), exact_solution, "numpy")

# plot the exact solution over Omega, for differing values of mu
N = 100
mus = [1, 10, 100]
xx = np.linspace(Omega[0], Omega[1], N)

def plot_exact(xx, mus):
    """plots exact solution 
    @param xx  np vector of spatial points at which to plot
    @param mus  list of values for mu
    """
    for mu in mus:
        plt.plot(xx, u_exact(xx,mu),label=r'$u^*,\, \mu = {}$'.format(mu))
    plt.title(r'exact solution $u^* = {}$'.format(ex_sol_latex), y=1.03)
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

plot_exact(xx, mus)

def a(w, v, mu0):
    """ symbolically evaluate the bilinear form a(w,v) """
    return integrate_domain( ddx(w)*ddx(v) + mu0*w*v )

# functional f(v)
def f(v): return 0

def Fi(wEn, psi, mu0):
    """ return Load vector entry """
    return f(psi) - a(wEn, psi, mu0)

def galerkin_approx(basis, wEn, mu0):
    """ computes the galerkin approximation to the Q1 problem
    @param basis  list of sympy expressions denoting the approx basis
    @param wEn  sympy expression of the lifting function
    """
    # assemble/solve linear system for galerkin coefficients
    n = len(basis)
    A, F = sym.zeros(n, n), sym.zeros(n, 1)
    for i in range(n):
        for j in range(n):
            psi_i, psi_j = basis[i], basis[j]
            A[i, j] = a(psi_j, psi_i, mu0)
        F[i,0] = Fi(wEn, psi_i, mu0)
    alpha = A.LUsolve(F)

    # build galerkin solution
    uG = 0
    for i, alpha_i in enumerate(alpha):
        uG += alpha_i * basis[i]
    uG += wEn
    return uG

uG_I = dict()
X_G, wEn = [x], 1
mu0s = [sym.Rational(1,10), 1, 10]
for mu0 in mu0s:
    uG_I[mu0] = galerkin_approx(X_G, wEn, mu0)
uG_I

def plot_galerkin_and_exact(uG, mu0s):
    """ plot a set of galerkin solutions vs exact """
    colors=sns.color_palette()
    for idx, mu0 in enumerate(mu0s):
        _ug = sym.lambdify(x, uG[mu0], "numpy")
        plt.plot(xx, _ug(xx), color=colors[idx],
                 label=r'$u_G,\, \mu = {}$'.format(mu0))
        plt.plot(xx, u_exact(xx, float(mu0)), '--')
    plt.legend(loc='lower left')
    plt.title(r'Galerkin solutions $u_G$ and exact solutions $u^*$')
    plt.show()

plot_galerkin_and_exact(uG_I, mu0s)

uG_II = dict()
mu0s = [sym.Rational(1,10), 1, 10]
for mu0 in mu0s:
    X_G, wEn = [x], sym.exp(-sym.sqrt(mu0)*x)
    uG_II[mu0] = galerkin_approx(X_G, wEn, mu0)
uG_II

plot_galerkin_and_exact(uG_II, mu0s)

uG_III = dict()
X_G, wEn = [x, x**2], 1
mu0s = [sym.Rational(1,10), 1, 10]
for mu0 in mu0s:
    uG_III[mu0] = galerkin_approx(X_G, wEn, mu0)
uG_III

plot_galerkin_and_exact(uG_III, mu0s)

def Pi(w, mu0):
    """ evaluates the pot. energy functional assoc. w/ the weak form
    @param w  symbolic expression w
    """
    return sym.Rational(1,2)*integrate_domain((ddx(w))**2 +mu0*(w**2))

def energy_norm(w, mu0):
    """ evaluates the energy norm of a function w"""
    return sym.sqrt( a(w, w, mu0) )
                      
def U(w, mu0):
    """evaluates the no-adjective energy functional """
    return a(w, w, mu0)

mu0s = [sym.Rational(1,10), 1, 10]
print('potential energy:')
print('{:12}{:10}{:10}'.format('mu', 'uG I', 'uG II'))
for mu0 in mu0s:
    _ugI = uG_I[mu0]
    _ugII = uG_II[mu0]
    print('{:<5} {:10.3} {:10.3}'.format(
        mu0, float(Pi(_ugI, mu0)), float(Pi(_ugII, mu0)) ))
 
print('\n')
print('no-adjective energy:')
print('{:12}{:10}{:10}'.format('mu', 'uG I', 'uG II'))
for mu0 in mu0s:
    _ugI = uG_I[mu0]
    _ugII = uG_II[mu0]
    print('{:<5} {:10.3} {:10.3}'.format(
        mu0, float(U(_ugI, mu0)), float(U(_ugII, mu0)) ))

