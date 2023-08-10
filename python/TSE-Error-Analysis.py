import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# to get LaTeX function rendereing:
sym.init_printing()

sym.var('x')

def plot_taylor(x0=0.0, n=1):
    func = sym.sin(x)#/x
    taylor = sym.series(func, x0=x0, n=n+1).removeO()

    evalfunc = sym.lambdify(x, func, modules=['numpy'])
    evaltaylor = sym.lambdify(x, taylor, modules=['numpy'])

    t = np.linspace(-2*np.pi, 3*np.pi, 100)
    plt.figure(figsize=(10,8))
    plt.plot(t, evalfunc(t), 'b', label='sin(x)')
    plt.plot(t, evaltaylor(t), 'r', label='Taylor')
    plt.plot(x0, evalfunc(x0), 'go', label='x0', markersize = 12)
    plt.legend(loc='upper left')
    plt.xlim([-1*np.pi, 2*np.pi])
    plt.ylim([-3,3])
    plt.show()

plot_taylor()

from ipywidgets import interactive
from IPython.display import Audio, display

plt.figure(figsize=(10,8))
plt.style.use("bmh")

v = interactive(plot_taylor, x0=(0.0,np.pi,np.pi/10.), n=(1,8), r=(2,4))
display(v)

def plot_taylor_lnx(x0=0.0, n=1):
    """Same method, different base function"""
    func = sym.ln(x)#/x
    taylor = sym.series(func, x0=x0, n=n+1).removeO()

    evalfunc = sym.lambdify(x, func, modules=['numpy'])
    evaltaylor = sym.lambdify(x, taylor, modules=['numpy'])

    t = np.linspace(0.01, 10, 100)
    plt.figure(figsize=(10,8))
    plt.plot(t, evalfunc(t), 'b', label='ln(x)')
    plt.plot(t, evaltaylor(t), 'r', label='Taylor')
    plt.plot(x0, evalfunc(x0), 'go', label='x0', markersize = 12)
    plt.legend(loc='upper left')
    plt.xlim([-0.2,10])
    plt.ylim([-3,3])
    plt.show()

v = interactive(plot_taylor_lnx, x0=(1,3,0.2), n=(1,8), r=(2,4))
display(v)

def plot_taylor_error(x0=0.0, n=1, s=1):
    func = sym.sin(x)#/x
    taylor = sym.series(func, x0=x0, n=n+1).removeO()

    evalfunc = sym.lambdify(x, func, modules=['numpy'])
    evaltaylor = sym.lambdify(x, taylor, modules=['numpy'])

    t = np.linspace(-2*np.pi, 3*np.pi, 100)
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(211)
    # plt.plot(t, evalfunc(t), 'b', label='sin(x)')
    ax1.plot(t, evaltaylor(t)-evalfunc(t), 'r', label='Taylor error')
    ax1.plot(x0, 0, 'go', label='x0', markersize = 12)
    plt.legend(loc='upper left')
    ax1.set_xlim([-1*np.pi, 2*np.pi])
    ax1.set_ylim([-s*3,s*3])
    ax2 = fig.add_subplot(212)
    ax2.plot(t, evalfunc(t), 'b', label='sin(x)')
    ax2.plot(t, evaltaylor(t), 'r', label='Taylor')
    ax2.plot(x0, evalfunc(x0), 'go', label='x0', markersize = 12)
    plt.legend(loc='upper left')
    ax2.set_xlim([-1*np.pi, 2*np.pi])
    ax2.set_ylim([-3,3])


    plt.show()

plot_taylor_error()

v = interactive(plot_taylor_error, x0=(0,np.pi,0.1*np.pi), n=(1,8), r=(2,4), s=(1,10))
display(v)





