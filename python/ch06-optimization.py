get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import numpy as np
import sympy
from scipy import optimize     # nonlinear optimization
import cvxopt                  # convex optimization

sympy.init_printing()

from __future__ import division

# example: area of cylinder with unit volume
# r = radius, h = height, f(r,h) = 2*pi*r^2 + 2*pi*r*h

#
# 2D optimization problem with an equality constraint
#

r, h = sympy.symbols("r, h")

Area   = 2*sympy.pi*r**2 + 2*sympy.pi*r*h
Volume =                     sympy.pi*r**2*h

h_r = sympy.solve(Volume - 1)[0]

Area_r = Area.subs(h_r)

rsol = sympy.solve(Area_r.diff(r))[0]
rsol

_.evalf()

# verify 2nd derivative is positive (rsol is a minimum)
Area_r.diff(r, 2).subs(r, rsol)

Area_r.subs(r, rsol)

_.evalf()

def f(r):
    return 2 * np.pi * r**2 + 2 / r

r_min = optimize.brent(f, brack=(0.1, 4))
r_min, f(r_min)

# radius that minimizes cylinder area ~ 0.54;
# corresponding min area ~ 5.54

optimize.minimize_scalar(f, bracket=(0.1, 5))

r = np.linspace(0, 2, 100)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(r, f(r), lw=2, color='b')
ax.plot(r_min, f(r_min), 'r*', markersize=15)
ax.set_title(r"$f(r) = 2\pi r^2+2/r$", fontsize=18)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_ylim(0, 30)

fig.tight_layout()
fig.savefig('ch6-univariate-optimization-example.pdf')

x1, x2 = sympy.symbols("x_1, x_2")

# objective function
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2

# gradient
fprime_sym = [f_sym.diff(x_) 
              for x_ in (x1, x2)]

sympy.Matrix(fprime_sym)

# hessian
fhess_sym = [
    [f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] 
    for x2_ in (x1, x2)]

sympy.Matrix(fhess_sym)

# use symbolic expressions to create vectorized functions for them
f_lmbda      = sympy.lambdify((x1, x2), f_sym, 'numpy')
fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')
fhess_lmbda  = sympy.lambdify((x1, x2), fhess_sym, 'numpy')

# funcs returned by sympy.lambdify take one arg for each var
# SciPy optimization func expect a vectorized function.
# need a wrapper.

def func_XY_X_Y(f):
    """
    Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda X: np.array(f(X[0], X[1]))

f      = func_XY_X_Y(f_lmbda)
fprime = func_XY_X_Y(fprime_lmbda)
fhess  = func_XY_X_Y(fhess_lmbda)

# optimize using (0,0) as a starting point
X_opt = optimize.fmin_ncg(f, (0, 0), fprime=fprime, fhess=fhess)

X_opt # minimum x1,x2

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, f_lmbda(X, Y), 50)
ax.plot(X_opt[0], X_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
fig.tight_layout()
fig.savefig('ch6-examaple-two-dim.pdf');

# objective function
def f(X):
    x, y = X
    return (4*np.sin(np.pi*x) + 6*np.sin(np.pi*y)) + (x-1)**2 + (y-1)**2

# brute-force search:
# slice objects == coordinate grid search space
# finish=None == auto-refine best candidate

x_start = optimize.brute(f, 
                         (slice(-3, 5, 0.5), 
                          slice(-3, 5, 0.5)), 
                         finish=None)
x_start, f(x_start)

# we now have good starting point for interative solver like BFGS
x_opt = optimize.fmin_bfgs(f, x_start)

x_opt, f(x_opt)

# visualize solution
# need wrapper to shuffle params

def func_X_Y_to_XY(f, X, Y):
    s = np.shape(X)
    return f(
        np.vstack(
            [X.ravel(), Y.ravel()])).reshape(*s)

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-3, 5, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 25)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
fig.tight_layout()
fig.savefig('ch6-example-2d-many-minima.pdf');

def f(x, beta0, beta1, beta2):
    return beta0 + beta1 * np.exp(-beta2 * x**2)

beta = (0.25, 0.75, 0.5)

# generate random datapoints
xdata = np.linspace(0, 5, 50)
y = f(xdata, *beta)
ydata = y + 0.05 * np.random.randn(len(xdata))

# start solver by defining function for residuals
def g(beta):
    return ydata - f(xdata, *beta)

# define initial guess for parameter vector
beta_start = (1, 1, 1)
# let leastsq() solve it
beta_opt, beta_cov = optimize.leastsq(g, beta_start)

# results
beta_opt

fig, ax = plt.subplots()

ax.scatter(xdata, ydata)
ax.plot(xdata, y, 'r', lw=2)
ax.plot(xdata, f(xdata, *beta_opt), 'b', lw=2)
ax.set_xlim(0, 5)
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)

fig.tight_layout()
fig.savefig('ch6-nonlinear-least-square.pdf')

# alternative method: curve_fit()
# convenience wrapper around leastsq()
# eliminates need to explicitly define residual function

beta_opt, beta_cov = optimize.curve_fit(f, xdata, ydata)
beta_opt

# objective function
def f(X):
    x, y = X
    return (x-1)**2 + (y-1)**2

x_opt = optimize.minimize(
    f, (0, 0), 
    method='BFGS').x

# boundary constraints
bnd_x1, bnd_x2 = (2, 3), (0, 2)

x_cons_opt = optimize.minimize(
    f, np.array([0, 0]), 
    method='L-BFGS-B', 
    bounds=[bnd_x1, bnd_x2]).x

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]), 
                           bnd_x1[1] - bnd_x1[0], bnd_x2[1] - bnd_x2[0],
                           facecolor="grey")
ax.add_patch(bound_rect)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)

fig.tight_layout()
fig.savefig('ch6-example-constraint-bound.pdf');

# unconstrained = blue star
# constrained   = red star

# use case:
# maximize volume of rectangle with dimensions x1,x2,x3
# constraint: total surface area must be unity (1.0)

# symbolics first
x = x1, x2, x3, l = sympy.symbols("x_1, x_2, x_3, lambda")

# volume
f = x1*x2*x3

# surface area constraint
g = 2 * (x1*x2 + x2*x3 + x3*x1) - 1

# Lagrangian
L      = f + l*g

# Lagrangian gradient
grad_L = [sympy.diff(L, x_) 
          for x_ in x]

# solve for zero. should return two points.
# However, 2nd point has x1<0 = not viable use case. (x1 is a dimension)
# so 1st point must be answer.
sols = sympy.solve(grad_L)
sols

# verify by eval'ing constraint func & objective func using answer
g.subs(sols[0]), f.subs(sols[0])

# objective function
def f(X):
    return -X[0] * X[1] * X[2]
# constraint function
def g(X):
    return 2 * (X[0]*X[1] + X[1] * X[2] + X[2] * X[0]) - 1

constraints = [dict(type='eq', fun=g)] # type = 'eq'

result = optimize.minimize(
    f, [0.5, 1, 1.5], 
    method='SLSQP', 
    constraints=constraints)
result

result.x

# objective function
def f(X):
    return (X[0] - 1)**2 + (X[1] - 1)**2
# constraint function
def g(X):
    return X[1] - 1.75 - (X[0] - 0.75)**4

x_opt = optimize.minimize(
    f, (0, 0), method='BFGS').x

constraints = [dict(type='ineq', fun=g)] # type = 'ineq'

x_cons_opt = optimize.minimize(
    f, (0, 0), 
    method='SLSQP', 
    constraints=constraints).x

# alternative solver:
# constrained optimization by linear approximation (COBYLA)

x_cons_opt = optimize.minimize(
    f, (0, 0), 
    method='COBYLA', 
    constraints=constraints).x

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)

ax.plot(x_, 1.75 + (x_-0.75)**4, 'k-', markersize=15)
ax.fill_between(x_, 1.75 + (x_-0.75)**4, 3, color="grey")
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)

ax.set_ylim(-1, 3)
ax.set_xlabel(r"$x_0$", fontsize=18)
ax.set_ylabel(r"$x_1$", fontsize=18)
plt.colorbar(c, ax=ax)

fig.tight_layout()
fig.savefig('ch6-example-constraint-inequality.pdf');

c = np.array([-1.0, 2.0, -3.0])

A = np.array([[ 1.0, 1.0, 0.0],
              [-1.0, 3.0, 0.0],
              [ 0.0, -1.0, 1.0]])

b = np.array([1.0, 2.0, 3.0])

# using cvxopt library
# has unique classes for matrices & vectors - can talk to NumPy
A_ = cvxopt.matrix(A)
b_ = cvxopt.matrix(b)
c_ = cvxopt.matrix(c)

sol = cvxopt.solvers.lp(c_, A_, b_)
sol

x = np.array(sol['x'])
x

sol['primal objective']

