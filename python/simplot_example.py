from __future__ import print_function, division

get_ipython().magic('matplotlib notebook')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import numpy as np
from scipy.integrate import odeint

from pint import UnitRegistry
UNITS = UnitRegistry()

# TODO: move these definitions into simplot.py

def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d

class Simplot:
    
    def __init__(self):
        self.figure_states = dict()
        
    def get_figure_state(self, figure=None):
        if figure is None:
            figure = plt.gca()
        
        try:
            return self.figure_states[figure]
        except KeyError:
            figure_state = FigureState()
            self.figure_states[figure] = figure_state
            return figure_state
    
SIMPLOT = Simplot()

class FigureState:
    
    def __init__(self):
        self.lines = dict()
        
    def get_line(self, style, kwargs):
        key = style, kwargs.get('color')
        
        try:
            return self.lines[key]
        except KeyError:
            line = self.make_line(style, kwargs)
            self.lines[key] = line
            return line
    
    def make_line(self, style, kwargs):
        underride(kwargs, linewidth=2, alpha=0.6)
        lines = plt.plot([], style, **kwargs)
        return lines[0]

def plot(*args, **kwargs):
    """Makes line plots.
    
    args can be:
      plot(y)
      plot(y, style_string)
      plot(x, y)
      plot(x, y, style_string)
    
    kwargs are the same as for pyplot.plot
    
    If x or y have attributes label and/or units,
    label the axes accordingly.
    
    """
    x = None
    y = None
    style = 'bo-'
    
    # parse the args the same way plt.plot does:
    # 
    if len(args) == 1:
        y = args[0]
    elif len(args) == 2:
        if isinstance(args[1], str):
            y, style = args
        else:
            x, y = args
    elif len(args) == 3:
        x, y, style = args
    
    # label the y axis
    label = getattr(y, 'label', 'y')
    units = getattr(y, 'units', 'dimensionless')
    plt.ylabel('%s (%s)' % (label, units))
    
    # label the x axis
    label = getattr(x, 'label', 'x')
    units = getattr(x, 'units', 'dimensionless')
    plt.xlabel('%s (%s)' % (label, units))
        
    #print(type(x))
    #print(type(y))
        
    figure = plt.gcf()
    figure_state = SIMPLOT.get_figure_state(figure)
    line = figure_state.get_line(style, kwargs)
    
    ys = line.get_ydata()
    ys = np.append(ys, y)
    line.set_ydata(ys)
    
    if x is None:
        xs = np.arange(len(ys))
    else:
        xs = line.get_xdata()
        xs = np.append(xs, x)
    
    line.set_xdata(xs)
    
    #print(line.get_xdata())
    #print(line.get_ydata())
    
    axes = plt.gca()
    axes.relim()
    axes.autoscale_view()
    figure.canvas.draw()
    
def newplot():
    plt.figure()
    
def labels(ylabel, xlabel, title=None, **kwargs):
    plt.ylabel(ylabel, **kwargs)
    plt.xlabel(xlabel, **kwargs)
    plt.title(title, **kwargs)

def slope_func(Y, t):
    y1, y2 = Y
    y1p = y2
    y2p = -(y2 + y1)
    return y1p, y2p

y_init = [1, 0]
y_init

slope_func(y_init, 0)

ts = np.arange(0, 15.0, 0.1)
type(ts)

def ode_solve_nondim(slope_func, y_init, ts):
    """
    """    
    y_mags = [(y.magnitude if isinstance(y, UNITS.Quantity) else y)
               for y in y_init]
    #print(y_mags)
        
    y_units = [(y.units if isinstance(y, UNITS.Quantity) else UNITS.dimensionless)
               for y in y_init]
    #print(y_units)
        
    # invoke the ODE solver
    asol = odeint(slope_func, y_mags, ts)
    
    cols = asol.transpose()
    res = [col * unit for col, unit in zip(cols, y_units)]
    
    return res

def ode_solve(slope_func, y_init, ts):
    """
    """
    # TODO: check that slope_func returns elements that have the right units
    y_init = np.asarray(y_init, dtype=object)
        
    # invoke the ODE solver
    asol = odeint(slope_func, y_init, ts)
    
    cols = asol.transpose()
    res = [col * unit for col, unit in zip(cols, y_units)]
    
    return res

y_init

position, velocity = ode_solve_nondim(slope_func, y_init, ts)

position.label = 'position'
position.units

velocity.label = 'velocity'
velocity.units

newplot()
plot(ts, position, 'b')



cm = UNITS.centimeter
g = UNITS.gram
kg = UNITS.kilogram
s = UNITS.second
N = UNITS.newton
m = UNITS.meter

mass = 0.1 * (kg)
k = 1 * (N / m)
c = 0 * (N / (m/s))

def slope_func(Y, t):
    y1, y2 = Y
    y1p = y2
    y2p = -(c*y2 + k*y1) / mass
    return y1p, y2p

def slope_func_nondim(Y, t):
    c = c.magnitude
    k = k.magnitude
    mass = mass.magnitude
    
    y1, y2 = Y
    y1p = y2
    y2p = -(c*y2 + k*y1) / mass
    return y1p, y2p

y_init = [0.01 * (m), 0 * (m/s)]
y_init

slope_func(y_init, 0)

ts = np.arange(0, 15.0, 0.1) * (s)
ts.label = 'time'
ts

def euler(slope_func, y_init, ts):
    print(slope_func.func_closure)
    for name in slope_func.func_closure:
        print(name)
    return
    
    
    asol = np.empty((len(ts), len(y_init)))
    hs = np.diff(ts) * ts.units

    units = [yi.units for yi in y_init]
    asol[0] = [yi.magnitude for yi in y_init]
    
    y = np.asarray(y_init, dtype=object)
    
    for i, h in enumerate(hs):
        slope = slope_func(y, ts[i])
        y = [yi + h * si for yi, si in zip(y, slope)]
        asol[i+1] = [yi.magnitude for yi in y]
    
    cols = asol.transpose()
    res = [col * unit for col, unit in zip(cols, units)]
    return res

position, velocity = euler(slope_func, y_init, ts)

newplot()
plot(ts, position, 'b')



newplot()

plot(1)

plot([2,3,4])

plot([4,5,6], 'gs-')

plot([7,8,9], color='red')

plot([1,2,3], [9,8,7], color='red')

newplot()

plot(1, 1, 'bo-')

plot(2, 2, 'bo')

plot(3, 2, 'bo-')

plot(3, 2, color='orange')

plot(2, 3, color='orange')

plot(1, 3)

labels('The y axis', 'The x axis', 'The title')

figure2 = newplot()

plot(1, 1)



newplot()

for i in range(10):
    plot(i, i, 'bo-')
    plot(i, 2*i, 'rs-')

newplot()

xs = np.arange(20)
ys = np.sin(xs)
plot(xs, ys)

plot(xs, ys+1, color='red')

plot(xs+20, ys)

a = 150
b = 150

a_to_b = round(0.05 * a) - round(0.03 * b)

a -= a_to_b
b += a_to_b

a, b

newplot()

a = 150
b = 150

for i in range(30):
    plot(i, a, color='red')
    plot(i, b, color='blue')
    a_to_b = round(0.05 * a) - round(0.03 * b)
    a -= a_to_b
    b += a_to_b

def f(u, t):
    print(u.units)
    return a*u*(1 - u/R)

a = 2
R = 1E+5 * (m)
A = 1 * (m)

import odespy
solver = odespy.RK4(f)
solver.set_initial_condition(A)

T = 10 # end of simulation
N = 30  # no of time steps
time_points = np.linspace(0, T, N+1) * (s)
u, t = solver.solve(time_points)

value = np.array(f(A, 0), dtype=object)
f(A, 0)
value

np.asarray(A)

from scipy.optimize import bisect as root_find

def f(x):
    print(type(x))
    return x**3 - x**2 - x - 1

a = 0 * (UNITS.meter)
b = 4 * (UNITS.meter)
res = root_find(f, a, b)

type(res)

get_ipython().magic('pinfo2 root_find')

x = 5

def foo():
    print(y)



x

def decorate(func):
    x = 7
    y = 3
    def func_wrapper():
        func()
    return func_wrapper

bar = decorate(foo)

bar.func_closure[0].cell_contents.func_name

bar()

def foo():
    x = 1
    def bar():
        y = 3
    print(bar.func_closure)

foo()



