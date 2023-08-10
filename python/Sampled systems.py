import numpy
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from ipywidgets import interact, Checkbox

f = numpy.sin

maxt = 100
t = numpy.linspace(0, maxt, 1000)
y = f(t)

def show_sampled(T=5.5, show_f=True):
    t_sampled = numpy.arange(0, maxt, T)
    y_sampled = f(t_sampled)

    if show_f:
        plt.plot(t, y)
    plt.scatter(t_sampled, y_sampled)
    plt.xlim(0, maxt)

interact(show_sampled, T=(0.1, 10), show_f=Checkbox());



