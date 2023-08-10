import numpy
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

zeros = [1]
poles = [-1 + 1j, -1 - 1j]
gain = 1

def polynomial(roots, s):
    retval = 1
    for r in roots:
        retval *= s - r
    return retval

def G(s):
    return gain*polynomial(zeros, s)/polynomial(poles, s)

def plotcomplex(curve, color='blue', marker=None):
    plt.plot(numpy.real(curve), numpy.imag(curve), color=color, marker=marker)

def plotpz():
    for p in poles:
        plotcomplex(p, color='red', marker='x')
    for z in zeros:
        plotcomplex(z, color='red', marker='o')

def fixaxis(size=5):
    """ Change to cross-style axes through the origin and fix size"""
    plt.axis([-size, size, -size, size])
    ax = plt.gca()
    # from http://stackoverflow.com/questions/25689238/show-origin-axis-x-y-in-matplotlib-plot
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    
    # Set axis to equal aspect ratio
    ax.set_aspect('equal')

from ipywidgets import interact

def plotsituation(contour):
    plotcomplex(contour)
    plotcomplex(G(contour), color='red')
    plotpz()
    fixaxis()

theta = numpy.linspace(0, 2*numpy.pi, 1000)

def argumentprinciple(centerreal=(-2., 2.), centerimag=(-2., 2.), radius=(0.5, 3)):
    contour = radius*numpy.exp(1j*theta) + centerreal + 1j*centerimag
    plotsituation(contour)    

interact(argumentprinciple)

omega = numpy.logspace(-2, 2, 1000)
Dcontour = numpy.concatenate([1j*omega, -1j*omega[::-1]]) # We're ignoring the infinite arc

K = 1

plotcomplex(K*G(Dcontour) + 1)
fixaxis(2)

def nyquistplot(K):
    plotcomplex(K*G(Dcontour))
    plotcomplex(-1, color='red', marker='o')
    fixaxis(size=2)

nyquistplot(K=1)

interact(nyquistplot, K=(0.5, 5.))

def bodeplot(K):
    plt.figure(figsize=(10,5))
    freqresp = K*G(1j*omega)
    plt.subplot(2, 2, 1)
    plt.loglog(omega, numpy.abs(freqresp))
    plt.ylim([0.1, 10])
    plt.axhline(1)
    plt.subplot(2, 2, 3)
    plt.semilogx(omega, numpy.unwrap(numpy.angle(freqresp)) - numpy.angle(freqresp[0])) # We know the angle should start at 0
    plt.axhline(-numpy.pi)
    plt.subplot(1, 2, 2)
    nyquistplot(K)

interact(bodeplot, K=(0.5, 5.))

