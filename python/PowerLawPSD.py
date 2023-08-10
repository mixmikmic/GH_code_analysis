#FOR INTERACTIVE PLOTTING, RUN IMPORT BLOCK THEN LAST BLOCK
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

import scipy as sp
import pylab as plt
from sklearn import linear_model
from scipy import signal
from IPython.html.widgets import interactive, FloatSliderWidget, fixed, interaction
from IPython.display import display

#parameter definition
N_neurons = 10000.              #number of poisson neurons
dt = 1.0/1000.                  #time step
t_end = 5                      #seconds
t = sp.linspace(0,t_end,int(t_end/dt)) #time vector
ran_block = sp.random.uniform(0,1,(N_neurons,len(t))); #random matrix, threshold to make spikes

FR = sp.array([1,10]);
lam = FR * dt;
LFP = sp.zeros([len(lam), len(t)])
#get LFP from summed spikes and plot
plt.figure()
for i in range(0,len(lam)):
    LFP[i,:] = sp.sum((ran_block<lam[i]).astype(int),0);
    plt.plot(t[0:1000],LFP[i,0:1000], label="FR = " + str(FR[i]))

plt.legend(); plt.xlabel("Time [s]"); 
plt.title('LFP signals with population firing rates of 1Hz (blue) and 10Hz (green)')

#calculate PSD and plot
Fa, Pxx = signal.welch(LFP,nperseg=1000, fs=1000)
plt.figure()
for i in range(0,len(lam)):
    plt.loglog(Fa,Pxx[i,:], label="FR = " + str(FR[i]))

plt.legend(); plt.xlabel("Freq [Hz]")
plt.title('PSD of Above LFP traces')

Fa, Pxx = signal.welch(LFP,nperseg=1000)
Fa = Fa*1000.
PpowLaw = 1e8*sp.power((Fa),-3) #1/f^2 powerlaw PSD
Ptot = PpowLaw+Pxx
fit_range = sp.arange(10,100)
plt.figure()
plt.loglog(Fa,PpowLaw, 'k', label="1/f")
for i in range(0,len(FR)):
    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(sp.log10(Fa[fit_range, sp.newaxis]), sp.log10(Ptot[i,fit_range]))
    fit_line = model_ransac.predict(sp.log10(Fa[1:,sp.newaxis]))

    #find line of intersection with baseline power law PSD
    intersect = Fa[sum(((sp.power(10,fit_line[:,0]) - PpowLaw[1:])<0).astype(int))+1] 

    plt.loglog(Fa, Ptot[i,:], label="1/f + firing rate = " + str(FR[i]))
    plt.loglog(Fa[1:],sp.power(10,fit_line), '--r', linewidth=2., label="Robust Fit")
    plt.loglog([intersect, intersect], plt.ylim(), 'k--')
    
    print "------- Firing rate = " + str(FR[i])
    print "Fitted slope = " , model_ransac.estimator_.coef_[0][0]
    print "Intersection Frequency = ", intersect, "Hz"

plt.legend();plt.show()

#INTERACTIVE PLOT
def changeFR(logFR):
    FR = sp.power(10,logFR)
    lam = FR * dt;
    #get LFP from summed spikes and plot
    LFP = sp.sum((ran_block<lam).astype(int),0);
    Fa, Pxx = signal.welch(LFP,nperseg=1000)
    Fa = Fa*1000.
    PpowLaw = 1e8*sp.power((Fa),-3) #1/f powerlaw PSD
    Ptot = PpowLaw+Pxx
    fit_range = sp.arange(10,100)
    plt.figure()
    plt.loglog(Fa,PpowLaw, 'k', label="1/f")
 
    #Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(sp.log10(Fa[fit_range, sp.newaxis]), sp.log10(Ptot[fit_range]))
    fit_line = model_ransac.predict(sp.log10(Fa[1:,sp.newaxis]))


    plt.loglog(Fa, Ptot, label="1/f + firing rate = " + str(FR))
    plt.loglog(Fa[1:],sp.power(10,fit_line), '--r', linewidth=2., label="Robust Fit")
    
    print "------- Firing rate = " + str(FR)
    print "Fitted slope = " , model_ransac.estimator_.coef_[0][0]
    plt.legend();plt.show()
    
 
slider = interaction.interact(changeFR,
                     logFR = FloatSliderWidget(min = -2, max = 2, step = 0.2, value = 0, msg_throttle=1 ))
display(slider)



