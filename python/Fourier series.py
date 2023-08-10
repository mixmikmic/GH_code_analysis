#NAME: Fourier Series
#DESCRIPTION: Creating a Fourier series for a function with period 2L.

get_ipython().magic('matplotlib notebook')
import numpy as np
import scipy.integrate as integrate
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
from matplotlib import animation

def generate_fourier_series(num_terms, from_func, limits):
    function_series = []
    func = from_func
    diff = limits[1] - limits[0]
    integral, __ = integrate.quad(func, limits[0], limits[1])
    a0 = 2 * integral / diff
    function_series.append(lambda x: a0/2)
    n = 1
    
    while n <= num_terms:
        cos_part = lambda x, n=n: func(x) * cos((2 * n * pi * x)/diff)
        sin_part = lambda x, n=n: func(x) * sin((2 * n * pi * x)/diff)
        integral, __ = integrate.quad(cos_part, limits[0], limits[1])
        an = 2 * integral / diff
        integral, __ = integrate.quad(sin_part, limits[0], limits[1])
        bn = 2 * integral / diff
        f1 = lambda x, an=an, bn=bn, n=n: an * cos((2 * n * pi * x)/diff) + bn * sin((2 * n * pi * x)/diff)
        function_series.append(f1)
        n += 1

    def fourier_func(x, evaluate_terms=None):
        to_return = 0.
        if not evaluate_terms:
            evaluate_terms = len(function_series)
        if evaluate_terms > len(function_series):
            evaluate_terms = len(function_series)
        terms_evaluated = 0
        while terms_evaluated < evaluate_terms:
            to_return += function_series[terms_evaluated](x)
            terms_evaluated += 1
        return to_return
    
    return fourier_func

def f(x):
    if x < 0.:
        return 1.
    else:
        return 0.

def animate(i):
    line.set_ydata(func(x, evaluate_terms=i))  
    return line,


def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,


limits = [-1., 1.]
num_terms = 100
func = generate_fourier_series(num_terms=num_terms, from_func=f, limits=limits)
x = np.arange(-1., 1., 0.01)
fig, ax = plt.subplots()
line, = ax.plot(x, func(x, evaluate_terms=0))
numframes = num_terms + 50
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=numframes, interval=20, blit=True)
plt.show()



