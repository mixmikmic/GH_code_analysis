# Import matplotlib (plotting) and numpy (numerical arrays).
# This enables their use in the Notebook.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import turtle


# Import IPython's interact function which is used below to
# build the interactive widgets
from IPython.html.widgets import interact


visc =  10**3 # [Pa s]
R_nozzle= 0.0015 #[m]
h = 0.0024 #[m]

def plot_flow_rate(D=0.05,L = 1, angle= 17*2*np.pi/360, N = 100/60,L_nozzle = 0.04,plot_nozzle=True):
    
    """
    plot nozzle variable can be set to false in order to not to show its plot.
    """
    R = D/2 #[m]
    AP = np.linspace(100, 700, 100)*10**6 #[Pa]
    Q = np.pi**2 * D**2 *(np.sin(angle)*np.cos(angle))*h*N/2 - h**3 * np.pi * D * np.sin(angle)**2 * AP/12/visc/L #[m3 s-1]

    
    Qn = np.pi * R_nozzle**4 / 8 / L_nozzle * AP/visc #[m3 s-1]


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('$\Delta P $ (Pa) ')
    ax.set_ylabel('Volumetric flow rate (m3 s)')
    ax.set_title('Extrusion stage operating lines')

    if plot_nozzle:
        ax.plot(AP, Qn, color='red', linestyle='solid', linewidth=2)
    
    ax.plot(AP,  Q,  marker='o', linewidth=2)


interact(plot_flow_rate, D=(0.01, 0.09, 0.01), L=(0.4,20,1), angle=(10*2*np.pi/360,30*2*np.pi/360,2*np.pi/360), N=(0.5,3,0.5),L_nozzle = (0.01,0.08,0.01), plot_nozzle=True);

