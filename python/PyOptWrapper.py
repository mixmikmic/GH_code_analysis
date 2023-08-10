from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, NLPQLP, ALPSO, SLSQP
import numpy as np

# ---- Rosenbrock with gradients, test different optimizers ------


def rosen(x):

    f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    c = []
    ceq = []

    gf = [2*(1 - x[0])*-1 + 200*(x[1] - x[0]**2)*-2*x[0], 200*(x[1] - x[0]**2)]
    gc = []
    gceq = []

    return f, c, ceq, gf, gc, gceq


x0 = [4.0, 4.0]
ub = [5.0, 5.0]
lb = [-5.0, -5.0]

optimizer = NSGA2()
optimizer.setOption('maxGen', 200)

xopt, fopt, info = optimize(rosen, x0, lb, ub, optimizer)
print 'NSGA2:', xopt, fopt, info


optimizer = SNOPT()

xopt, fopt, info = optimize(rosen, x0, lb, ub, optimizer)
print 'SNOPT:', xopt, fopt, info


optimizer = NLPQLP()

xopt, fopt, info = optimize(rosen, x0, lb, ub, optimizer)
print 'NLPQLP:', xopt, fopt, info


optimizer = SLSQP()

xopt, fopt, info = optimize(rosen, x0, lb, ub, optimizer)
print 'SLSQP:', xopt, fopt, info


optimizer = ALPSO()

xopt, fopt, info = optimize(rosen, x0, lb, ub, optimizer)
print 'ALPSO:', xopt, fopt, info

# ---- Rosenbrock with no gradients ------

def rosen(x):

    f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    c = []
    ceq = []

    return f, c, ceq



optimizer = SNOPT()

xopt, fopt, info = optimize(rosen, x0, lb, ub, optimizer)
print 'SNOPT:', xopt, fopt, info

# ---- test linear constraints ------


H = np.array([[1.0, -1.0], [-1.0, 2.0]])
f = np.array([-2.0, -6.0])
A = np.array([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
b = np.array([2.0, 2.0, 3.0])


def obj(x):
    ff = 0.5*np.dot(x, np.dot(H, x)) + np.dot(f, x)
    c = []
    ceq = []

    return ff, c, ceq

x0 = [4.0, 4.0]
ub = [100.0, 100.0]
lb = [0.0, 0.0]


optimizer = SNOPT()

xopt, fopt, info = optimize(obj, x0, lb, ub, optimizer, A, b)
print 'linear constraints:', xopt, fopt, info

# ------ constrained problem ---------

from math import exp

def barnes(x):

    a1 = 75.196
    a3 = 0.12694
    a5 = 1.0345e-5
    a7 = 0.030234
    a9 = 3.5256e-5
    a11 = 0.25645
    a13 = 1.3514e-5
    a15 = -5.2375e-6
    a17 = 7.0e-10
    a19 = -1.6638e-6
    a21 = 0.0005
    a2 = -3.8112
    a4 = -2.0567e-3
    a6 = -6.8306
    a8 = -1.28134e-3
    a10 = -2.266e-7
    a12 = -3.4604e-3
    a14 = -28.106
    a16 = -6.3e-8
    a18 = 3.4054e-4
    a20 = -2.8673

    x1 = x[0]
    x2 = x[1]
    y1 = x1*x2
    y2 = y1*x1
    y3 = x2**2
    y4 = x1**2

    # --- function value ---

    f = a1 + a2*x1 + a3*y4 + a4*y4*x1 + a5*y4**2 +         a6*x2 + a7*y1 + a8*x1*y1 + a9*y1*y4 + a10*y2*y4 +         a11*y3 + a12*x2*y3 + a13*y3**2 + a14/(x2+1) +         a15*y3*y4 + a16*y1*y4*x2 + a17*y1*y3*y4 + a18*x1*y3 +         a19*y1*y3 + a20*exp(a21*y1)

    # --- constraints ---

    c = np.zeros(3)
    c[0] = 1 - y1/700.0
    c[1] = y4/25.0**2 - x2/5.0
    c[2] = (x1/500.0- 0.11) - (x2/50.0-1)**2


    # --- derivatives of f ---

    dy1 = x2
    dy2 = y1 + x1*dy1
    dy4 = 2*x1
    dfdx1 = a2 + a3*dy4 + a4*y4 + a4*x1*dy4 + a5*2*y4*dy4 +         a7*dy1 + a8*y1 + a8*x1*dy1 + a9*y1*dy4 + a9*y4*dy1 + a10*y2*dy4 + a10*y4*dy2 +         a15*y3*dy4 + a16*x2*y1*dy4 + a16*x2*y4*dy1 + a17*y3*y1*dy4 + a17*y3*y4*dy1 + a18*y3 +         a19*y3*dy1 + a20*exp(a21*y1)*a21*dy1

    dy1 = x1
    dy2 = x1*dy1
    dy3 = 2*x2
    dfdx2 = a6 + a7*dy1 + a8*x1*dy1 + a9*y4*dy1 + a10*y4*dy2 +         a11*dy3 + a12*x2*dy3 + a12*y3 + a13*2*y3*dy3 + a14*-1/(x2+1)**2 +         a15*y4*dy3 + a16*y4*y1 + a16*y4*x2*dy1 + a17*y4*y1*dy3 + a17*y4*y3*dy1 + a18*x1*dy3 +         a19*y3*dy1 + a19*y1*dy3 + a20*exp(a21*y1)*a21*dy1

    dfdx = np.array([dfdx1, dfdx2])


    # --- derivatives of c ---

    dcdx = np.zeros((3, 2))
    dcdx[0, 0] = -x2/700.0
    dcdx[0, 1] = -x1/700.0
    dcdx[1, 0] = 2*x1/25**2
    dcdx[1, 1] = -1.0/5
    dcdx[2, 0] = 1.0/500
    dcdx[2, 1] = -2*(x2/50.0-1)/50.0

    # dcdx = np.transpose(dcdx)  # matlab format

    return f/30.0, c, [], dfdx/30.0, dcdx, []

x0 = np.array([10.0, 10.0])
lb = np.array([0.0, 0.0])
ub = np.array([65.0, 70.0])

optimizer = SNOPT()
xopt, fopt, info = optimize(barnes, x0, lb, ub, optimizer)
print 'barnes:', xopt, fopt, info

# --------- multiobjective ----------

def multiobj(x):

    f = np.zeros(2)
    f[0] = (x[0]+2)**2 + (x[1]+2)**2 - 10
    f[1] = (x[0]-2)**2 + (x[1]-2)**2 + 20

    c = [x[0] - 1]
    ceq = []

    return f, c, ceq

x0 = [4.0, 4.0]
ub = [20.0, 20.0]
lb = [-20.0, -20.0]

optimizer = NSGA2()
optimizer.setOption('maxGen', 200)

xopt, fopt, info = optimize(multiobj, x0, lb, ub, optimizer)
print 'NSGA2:'
print 'xopt =', xopt
print 'fopt =', fopt
print 'info =', info

def rosenarg(x, a, b):  # note two extra arguments, you can add as many as you want, and they can be whatever type

    f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2 + a + b
    c = []
    ceq = []

    return f, c, ceq


x0 = [4.0, 4.0]
ub = [5.0, 5.0]
lb = [-5.0, -5.0]

optimizer = SNOPT()

xopt, fopt, info = optimize(rosenarg, x0, lb, ub, optimizer, args=(3.0, 4.0))  # pass the values for args here in a tuple
print 'args:', xopt, fopt, info

def barneseq(x):

    f, c, [], gf, gc, [] = barnes(x)

    x1 = x[0]
    x2 = x[1]

    ceq = np.zeros(2)
    ceq[0] = x1**2 + x2**2 - 3000
    ceq[1] = x1 + 2*x2 - 120

    return f, c, ceq

x0 = np.array([10.0, 10.0])
lb = np.array([0.0, 0.0])
ub = np.array([65.0, 70.0])

optimizer = SNOPT()
xopt, fopt, info = optimize(barneseq, x0, lb, ub, optimizer)
print 'barneseq:', xopt, fopt, info

def barneseqgrad(x):

    f, c, [], gf, gc, [] = barnes(x)

    x1 = x[0]
    x2 = x[1]

    ceq = np.zeros(2)
    ceq[0] = x1**2 + x2**2 - 3000
    ceq[1] = x1 + 2*x2 - 120

    gceq = np.zeros((2, 2))
    gceq[0, 0] = 2*x1
    gceq[0, 1] = 2*x2
    gceq[1, 0] = 1.0
    gceq[1, 1] = 2.0

    return f, c, ceq, gf, gc, gceq

x0 = np.array([10.0, 10.0])
lb = np.array([0.0, 0.0])
ub = np.array([65.0, 70.0])

optimizer = SNOPT()
xopt, fopt, info = optimize(barneseqgrad, x0, lb, ub, optimizer)
print 'barneseqgrad:', xopt, fopt, info



