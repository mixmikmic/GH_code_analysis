get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from scipy.optimize import minimize

from mpl_toolkits.mplot3d import *
from matplotlib import cm
from ipywidgets import interact

alpha = 0.6
p = 1
I = 100

# useful for plots and calculations
def budgetc(c0,p,I):
    '''c1 as a function of c0 along budget line'''
    return I - p*c0

def U(c, a=alpha):
    '''Utility at c=(c[0], c[1])'''
    return (c[0]**a)*(c[1]**(1-a))

def MU0(c, a=alpha):
    '''MU of Cobb-Douglas'''
    return  a*U(c,a)/c[0] 

def MU1(c, a=alpha):
    return  (1-a)*U(c,a)/c[1]

def indif(c0, ubar, a=alpha):
    '''c1 as function of c0, implicitly defined by U(c0, c1) = ubar'''
    return (ubar/(c0**a))**(1/(1-a))

def ademands(p,I,a =alpha):
    '''Analytic solution for interior optimum'''
    c0 = a * I/p
    c1 = (1-a)*I
    c = [c0,c1]
    uopt = U(c,a)
    return c, uopt

pmin, pmax = 1, 4
Imin, Imax = 10, 200
cmax = (3/4)*Imax/pmin
c0 = np.linspace(0.1,cmax,num=100)

def consume_plot(p, I, a=alpha):
    ce, uebar = ademands(p, I, a)
    fig, ax = plt.subplots(figsize=(9,9))
    ax.plot(c0, budgetc(c0, p, I), lw=2.5)
    ax.fill_between(c0, budgetc(c0, p, I), alpha = 0.2)
    ax.plot(c0, indif(c0, uebar, a), lw=2.5)
    ax.vlines(ce[0],0,ce[1], linestyles="dashed")
    ax.hlines(ce[1],0,ce[0], linestyles="dashed")
    ax.plot(ce[0],ce[1],'ob')
    ax.set_xlim(0, cmax)
    ax.set_ylim(0, cmax)
    ax.set_xlabel(r'$c_0$', fontsize=16)
    ax.set_ylabel('$c_1$', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

interact(consume_plot, p=(0.5,2,0.1), I=(50,150,10), a=(0.1,0.9,0.1));

def arb_plot(c0g):
    cg = [c0g,I-c0g]
    '''Display characteristics of a guess along the constraint'''
    fig, ax = plt.subplots(figsize=(9,9))
    ax.plot(c0, budgetc(c0, p, I), lw=1)
    ax.fill_between(c0, budgetc(c0, p, I), alpha = 0.2)
    ax.plot(c0, indif(c0, U(cg)), lw=2.5)
    ax.vlines(cg[0],0,cg[1], linestyles="dashed")
    ax.hlines(cg[1],0,cg[0], linestyles="dashed")
    ax.plot(cg[0],cg[1],'ob')
    mu0pd, mu1pd = MU0(cg), MU1(cg)/p
    if mu0pd > mu1pd:
        inq = r'$>$'
    elif mu0pd < mu1pd:
        inq = r'$<$'
    else:
        inq =r'$=$'
    ax.text(60, 120, r'$\frac{MU_0}{p_0}$'+inq+r'$\frac{MU_1}{p_1}$',fontsize=20)
    utext = r'$({:5.1f}, {:5.1f}) \ \ U={:5.3f}$'.format(cg[0], cg[1], U(cg))
    ax.text(60, 100, utext, fontsize=12)
    ax.set_xlim(0, cmax)
    ax.set_ylim(0, cmax)
    ax.set_xlabel(r'$c_0$', fontsize=16)
    ax.set_ylabel('$c_1$', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('The No-Arbitrage argument')
    plt.show()

interact(arb_plot, c0g=(1,I,0.1));

from scipy.optimize import minimize

def demands(p, I, a= alpha):
    
    def budget(x):
        return np.array([p*x[0] + x[1] - I])
    
    def negU(x):
        return -U(x, a)

    x = np.array([(I/2)/p, (I/2)])  # initial guess - spend half I on each good
    
    ux = minimize(negU, x, constraints = ({'type': 'eq', 'fun' : budget }) )
    
    return ux.x
    

[x0opt, x1opt] = demands(1, 100, 0.4)

x0opt, x1opt, U([x0opt, x1opt], 0.4)

ademands(1,100, 0.4)

interact(demands, p=(0.2,2,0.1), I = (50, 200, 10), a =(0.2, 0.8, 0.1));

