get_ipython().magic('matplotlib inline')
import numpy as np
from scipy import integrate
import nitime.algorithms as tsa
import nitime.utils as utils
from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate
import seaborn as sns
sns.set_palette("Dark2")  

# define model parameters
Tscale = 7.5 # in Kelvins
tscale = 1/6.0  # in years [=2 months]
hscale = 150.0 # in meters

def recharge_deriv((T,h),t, mu, en, c = 1.0,r= 0.25,alpha = 0.125, b0 = 2.5, gamma = 0.75):
    """Compute the time-derivative of the recharge system system."""
    b = b0*mu # coupling coefficient     
    R = gamma*b-c;  #Bjerknes coefficient
    
    return [R*T + gamma*h - en*(h + b*T)**3, -r*h - alpha*b*T]    

def recharge_solver(mu=0.7,en=0,x0 = [[0.1,-0.1]],max_time=500*6):    
    t = np.linspace(0, max_time, int(4*max_time))
    x_t = np.asarray([integrate.odeint(recharge_deriv, x0i, t, (mu, en))
                      for x0i in x0])
    return t, x_t

def recharge_plotter(t,x_t,mu,en,x0,last_time):       
    T = x_t[0,:,0]; h = x_t[0,:,1]
    ts = t*tscale  
    # obtain spectrum
    f_T, S_T, nu = tsa.multi_taper_psd(T,Fs=24.0)
    # compute average period
    DominantPeriod =  "{:4.2f}".format(1/f_T[np.argmax(S_T)])  
    # plot
    plt.close('all')  
    fig = plt.figure(figsize=(8,8))
    # plot traces
    ax1 = plt.subplot2grid((2, 2), (0, 0),colspan = 2)
    hdl = ax1.plot(ts[-last_time:],T[-last_time:],ts[-last_time:],h[-last_time:])
    ax1.legend(hdl,[r'$T_E$',r'$h_W$'],loc = 'upper left')
    ax1.set_title(r'Recharge oscillator solution, $\mu$ = %s, $e_n$ = %s'%(mu, en),fontsize=14)
    # phase portrait
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.plot(x_t[0,:,0],x_t[0,:,1])
    ax2.plot(x0[0][0],x0[0][1],'ro')
    ax2.set_xlabel(r'$T_E$'); ax2.set_ylabel(r'$h_W$')
    ax2.set_title("Phase portrait",fontsize=14)
    # 
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.loglog(f_T,S_T, label=r'$T_E$',color='black')
    ax3.legend()
    ax3.set_title(r'Spectrum, main peak @ %s years'%DominantPeriod,fontsize=14)
    ax3.set_xlabel(r'Period (Years)'); ax3.set_xlim([1e-2,6.0])
    ax3.set_ylabel(r'PSD')
    per = [50,20,10,5,2,1,0.5]
    xt = 1.0/np.array(per)
    ax3.set_xticks(xt)
    ax3.set_xticklabels(map(str, per))

mu, en = 0.65, 0  # set variable parameters
x0 = [[0.1,-0.1]]   # initial conditions
t, x_t = recharge_solver(mu,en)  # solve the system
nt = t.shape[0];  last_time = nt  # plot only the last "last_time" steps in case the plot is too dense
recharge_plotter(t, x_t, mu, en, x0, last_time) # plot it

mu, en = 2./3, 0  # set variable parameters
x0 = [[0.1,-0.1]]   # initial conditions
t, x_t = recharge_solver(mu,en)  # solve the system
nt = t.shape[0];  last_time = int(nt/10)
recharge_plotter(t, x_t, mu, en, x0, last_time) # plot it

# run the system for mu = 0.7, say

mu, en = 0.7, 3.0  # set variable parameters
x0 = [[0.1,-0.1]]   # initial conditions
t, x_t = recharge_solver(mu,en)  # solve the system
nt = t.shape[0];  last_time = int(nt/5)
recharge_plotter(t, x_t, mu, en, x0, last_time) # plot it

def recharge_deriv((T,h),t, mu, en, Af, Pf, c = 1.0,r= 0.25,alpha = 0.125, b0 = 2.5, gamma = 0.75):
    """Compute the time-derivative of the recharge system system."""
    b = b0*mu # coupling coefficient     
    R = gamma*b-c;  #Bjerknes coefficient
    forcing = Af*np.sin(2*np.pi*t/Pf)
    return [R*T + gamma*h - en*(h + b*T)**3 + forcing, -r*h - alpha*b*T]    
    

def recharge_solver(mu,en,x0, Af=1.0, Pf=6.0,max_time = 3000):  
    """ Solve recharge oscillator ODE system
        mu = effective coupling
        en = strength of nonlinear feedback
        x0 = initial conditions
        Af = forcing amplitude
        Pf = forcing period
    """
    # define time axis
    t = np.linspace(0, max_time, int(4*max_time),endpoint=False)
    # Solve for the trajectories
    x_t = np.asarray([integrate.odeint(recharge_deriv, x0i, t, (mu, en, Af, Pf))
                      for x0i in x0])
    return t, x_t
    
def recharge_plotter(T, h, t, mu, en, Fs, Af=0.0, Pf=6.0):
    """ Plot the output of recharge_solver 
        Fs = sampling frequency    
    """
    # extract output
    ts = t*tscale; nt = t.shape[0]
    last_time = np.round(nt/5.0)
    # obtain spectrum
    f_T, S_T, nu = tsa.multi_taper_psd(T,Fs)
    # compute average period
    DominantPeriod =  "{:4.2f}".format(1.0/f_T[np.argmax(S_T)])  
    # PLOT IT OUT
    plt.close('all')  
    fig = plt.figure(figsize=(8,8))
    # plot traces
    ax1 = plt.subplot2grid((2, 2), (0, 0),colspan = 2)
    hdl = ax1.plot(ts[-last_time:],T[-last_time:],ts[-last_time:],h[-last_time:])
    ax1.legend(hdl,[r'$T_E$',r'$h_W$'],loc = 'upper left')
    ax1.set_title(r'Recharge oscillator solution, $\mu$ = %s, $e_n$ = %s, $A_f=$ %s'%(mu, en, Af),fontsize=14)
    # phase portrait
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.plot(x_t[0,:,0],x_t[0,:,1])
    ax2.plot(x0[0][0],x0[0][1],'ro')
    ax2.set_xlabel(r'$T_E$'); ax2.set_ylabel(r'$h_W$')
    ax2.set_title("Phase portrait",fontsize=14)
    # 
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.loglog(f_T,S_T, label=r'$T_E$',color='black')
    ax3.tick_params(axis="both", which="both", bottom="on", top="off",  
                    labelbottom="on", left="on", right="off", labelleft="on",direction="out")  
    #ax3.minorticks_off()
    ax3.legend()
    ax3.set_title(r'T spectrum, main peak @ %s years'%DominantPeriod,fontsize=14)
    ax3.set_xlabel(r'Period (Years)'); ax3.set_xlim([1e-2,6.0])
    ax3.set_ylabel(r'PSD')
    per = [50,20,10,5,2,1]
    xt = 1.0/np.array(per)
    ax3.set_xticks(xt)
    ax3.set_xticklabels(map(str, per))
    

mu, en = 0.7, 3  # set free oscillator parameters
x0 = np.float64([[.1,-.1]]) # set initial conditions
Af = 0.01 # forcing amplitude
Pf = 6.0  # forcing period (in nondimensional units: 6 = 12 months = 1 year)

# solve system
t, x_t = recharge_solver(mu,en,x0,Af,Pf)
T = x_t[0,:,0]; h = x_t[0,:,1]
ts = t*tscale;
Fs = np.shape(np.where(ts<=1))[1] # get sampling frequency

# PLOT IT
recharge_plotter(T, h ,t, mu, en, Fs, Af)



