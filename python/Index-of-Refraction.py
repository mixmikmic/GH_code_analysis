import matplotlib.pyplot as plt
from IPython.display import Latex
from IPython.html.widgets import interactive
get_ipython().magic('pylab inline --no-import-all')
#If the widger sliders don't work, then you have been affected by the 'Big Split'
#If you have installed Ipython from the Anaconda distribution and your notebook
#does not have Jupyter logo at the top go to a terminal and type conda install jupyter
#If you do have jupyter installed try starting Ipython from a terminal by typing
# jupyter notebook then opening this notebook

from scipy.integrate import odeint, ode
from scipy import sqrt, linspace, cos, sin

def dy(y,t,gamma,omega_o,omega,amp):
    x,p=y[0],y[1]
    dx=p
    dp=-gamma*omega_o*p-omega_o**2*x+amp*cos(omega*t)
    return[dx,dp]

y0=[1.0,0.0]
t=linspace(0.,20,1000)

def solve_ODE(omega_o,gamma, omega, amp):
    y=odeint(dy,y0,t,args=(gamma,omega_o,omega,amp))
    field=[]
    field=amp*cos(omega*t)
    rcdef = plt.rcParams.copy()
    newparams = {'axes.labelsize': 14, 'axes.linewidth': 1, 'savefig.dpi': 300, 
             'lines.linewidth': 1.5, 'figure.figsize': (8, 4),
             'figure.subplot.wspace': 0.4,
             'ytick.labelsize': 12, 'xtick.labelsize': 12,
             'ytick.major.pad': 5, 'xtick.major.pad': 5,
             'legend.fontsize': 12, 'legend.frameon': False, 
             'legend.handlelength': 1.5}
    # Update the global rcParams dictionary with the new parameter choices
# Before doing this, we reset rcParams to its default again, just in case
    plt.rcParams.update(rcdef)
    plt.rcParams.update(newparams)
   
    
    fig,ax1 =plt.subplots()
    ax1.plot(t,y[:,0],'b-',label='r')
    ax1.set_xlabel('time (s)')

    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2=ax1.twinx()
    ax1.set_ylabel('displacement')
    ax2.set_ylabel('field')
    ax2.plot(t,field,'r-',label='driving field')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
       
    plt.rcParams.update(rcdef)

w=interactive(solve_ODE,omega_o=(1.,10.),gamma=(0.,10.),omega=(0.,20.),amp=(0.,10.))
display(w)

def complex_index(omega_o=10., gamma=1., omega_p=1):
    #create the "x-axis"
    omega=linspace(0,30,1000) 
    
    #Equation 
    nk=sqrt(1+(pow(omega_p,2))/(pow(omega_o,2)-omega*gamma*1.j-pow(omega,2)))
    n=nk.real
    kappa=nk.imag
    
    rcdef = plt.rcParams.copy()
    newparams = {'axes.labelsize': 14, 'axes.linewidth': 1, 'savefig.dpi': 300, 
             'lines.linewidth': 1.5, 'figure.figsize': (16, 6),
             'figure.subplot.wspace': 0.4,
             'ytick.labelsize': 12, 'xtick.labelsize': 12,
             'ytick.major.pad': 5, 'xtick.major.pad': 5,
             'legend.fontsize': 12, 'legend.frameon': False, 
             'legend.handlelength': 1.5}

    # Update the global rcParams dictionary with the new parameter choices
    # Before doing this, we reset rcParams to its default again, just in case
    plt.rcParams.update(rcdef)
    plt.rcParams.update(newparams)

    # Make the new figure with new formatting
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(omega, n, label ='index of refraction, n')
    axes[1].plot(omega, kappa, label='attenuation, \kappa')

    axes[0].legend()
    axes[1].legend()
    
    axes[0].set_xlabel(r'$\omega (\mathrm{s})^{-1}$')

    axes[1].set_xlabel(r'$\omega (\mathrm{s})^{-1}$')

    plt.rcParams.update(rcdef)

w=interactive(complex_index,omega_o=(1.,50.), gamma=(0,5.), omega_p=(1.,10.))
display(w)







