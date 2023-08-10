import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')
import ipywidgets as wdg
import IPython.display as display

def b_value(g, delta, Delta, gamma=42.576):
    """ 
    Calculate the b value
    
    Parameters
    ----------
    g : gradient strength (mT/m, typically around 40)
    delta : gradient duration
    Delta : diffusion duration
    gamma : the gyromagnetic ration (42.576 MHz/T for Hydrogen)
    
    """
    G = g*1e-3*1e-6 #convert to T/um
    gamma = 2*np.pi*gamma*1e6*1e-3 # convert to 1/ms/T (Hz = cycles/sec, 1 cycle = 2pi = 2pi/sec)
    b = gamma ** 2 * G ** 2 * delta ** 2 * (Delta-delta/3) # millisecons/micrometer^2  
    return 1000 * b #s/mm^2

def ST_equation(b, D, S0=1):
    """ 
    The Stejskal Tanner equation
    """
    return S0 * np.exp(-b * D)

def viz_gradients(g = 40, delta = 13, Delta = 60):
    S0 = 1
    t = np.arange(0, 10 + delta + Delta + 10, 1) # In msec
    grad = np.zeros(t.shape)
    grad[np.where(np.logical_and(t>10, t<10 + delta))] = g
    grad[np.where(np.logical_and((t>10 + Delta), t <(10+Delta + delta)))] = g
    b = b_value(g, delta, Delta)
    fig, ax = plt.subplots(1, 2, tight_layout=True)
    ax[0].plot(t, grad)
    ax[0].plot([10, 10+delta], [g+10, g+10],'k-')
    ax[0].plot([10], [g+10],'k<')
    ax[0].plot([10+delta], [g+10],'k>')
    ax[0].text(10+delta/2., g+20, '$\delta$')
    ax[0].plot([10, 10+Delta], [g+40, g+40],'k-')
    ax[0].plot([10], [g+40],'k<')
    ax[0].plot([10+Delta], [g+40],'k>')
    ax[0].text(10+Delta/2., g+60, '$\Delta$')
    
    ax[0].plot([10+Delta+delta+5, 10+Delta+delta+5], [10, g],'k-')
    ax[0].plot([10+Delta+delta+5], [10],'kv')
    ax[0].plot([10+Delta+delta+5], [g],'k^')
    ax[0].text(10+Delta+delta+5+5, g/2., 'g')
    ax[0].set_ylabel('Gradient amplitude(mT/m)')
    ax[0].set_xlabel('Time (msec)')
    ax[0].set_xlim([-10, max(t) + 10])
    ax[0].set_ylim([-10, 375])
    D = np.arange(0, 4, 0.01)
    ax[1].plot(D, ST_equation(b/1000., D, S0=S0))
    ax[1].plot([3.0, 3.0], [0, S0], '--k')
    ax[1].set_xlabel(r'D ($\frac{mm^2}{s})$')
    ax[1].set_ylabel('MRI signal')
    ax[1].text(1, 0.8, 'b=%s'%int(b))

vg_widget = wdg.interactive(viz_gradients, 
                            g=wdg.FloatSlider(min=10, max=300, step=10.0, value=40),
                            delta=wdg.FloatSlider(min=10, max=40, step=1.0, value=13),
                            Delta=wdg.FloatSlider(min=10, max=300, step=10.0, value=60)
                            )
display.display(vg_widget)

fig, ax = plt.subplots(1)
D = np.arange(0, 4, 0.04)
S0=1
b=1000
for b in [1000, 2000, 4000]:
    ST = ST_equation(b/1000., D, S0=S0)
    ax.plot(D, ST, label='%s'%b)
    ax.plot([3.0, 3.0], [0, S0], '--k')
    ax.set_xlabel(r'D ($\frac{mm^2}{s})$')
    ax.set_ylabel('$\\frac{\delta S}{\delta D} $')
plt.legend()

fig, ax = plt.subplots(1)
D = np.arange(0, 4, 0.04)
S0=1
b=1000
for b in [1000, 2000, 4000]:
    ST = ST_equation(b/1000., D, S0=S0)
    ax.plot(D[:-1], -np.diff(ST), label='%s'%b)
    ax.plot([3.0, 3.0], [0, S0], '--k')
    ax.set_xlabel(r'D ($\frac{mm^2}{s})$')
    ax.set_ylabel('$\\frac{\delta S}{\delta D} $')
    ax.set_ylim([0,0.15])
plt.legend()



