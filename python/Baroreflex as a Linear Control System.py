get_ipython().magic('matplotlib inline')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import control.matlab as control

sns.set_context('talk')

# needed to avoid a deprecation warning in the control library 
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

Kp = 1.16                  # mmHg/%SNA
tau = 1.0/(2*0.089*np.pi)  # sec
zeta = 1.23
td = 0.476

Gp_ = control.tf([Kp],[tau**2,2*zeta*tau,1])
print('\nWithout Time Delay -->\n', Gp_)

num,den = control.pade(td,3)
Gp = Gp_ * control.tf(num,den)
print('\nWith a 3rd order Pade approximation for Time Delay -->\n', Gp)

t = np.linspace(0,20,1000)
y,t = control.step(Gp,t)
plt.plot(t,y)
plt.xlabel('Time [sec]')
plt.ylabel('arterial pressure [mmHg]')
plt.title('Response of the Peripheral Arc to a 1% change in SNA');

w = np.logspace(-2,1,300)
mag,phase,w = control.bode(Gp,w)

def PID(Kc=1,tauI=0,tauD=0):
    alpha = 0.1
    Gc = control.tf([1],[1])
    if tauI != 0:
        Gc += control.tf([1],[1,0])
    if tauD != 0:
        Gc += control.tf([tauD,0],[alpha*tauD,1])
    return Kc*Gc

def P(Kc=1):
    return PID(Kc,0,0)

def PI(Kc=1,tauI=0):
    return PID(Kc,tauI,0)

def PD(Kc=1,tauD=0):
    return PID(Kc,0,tauD)

PID(1,1,1)

gm,pm,Wcg,Wcp = control.margin(Gp)
Kcu = gm
Pu = 2*np.pi/Wcp

print('Ultimate Gain = ', Kcu)
print('Ultimate Period = ', Pu)

def stepResponse(Gp,Gc,t):
    y,t = control.step(1/(1+Gp*Gc),t)
    u,t = control.step(-Gc/(1+Gp*Gc),t)
    plt.subplot(2,1,1)
    plt.plot(t,y,[0,0,max(t)],[0,1,1],'r--')
    plt.xlabel('Time [sec]')
    plt.ylabel('Arterial Pressure [mmHg]')
    plt.title('Baroreflex Disturbance Response')
    plt.legend(['Arterial Pressure','Disturbance'],loc='lower left')
    plt.subplot(2,1,2)
    plt.plot(t,u)
    plt.ylim(-10,10)
    plt.ylabel('Sympathetic Neural Activity [%]')
    plt.tight_layout()
         
t = np.linspace(0,15,1000)
stepResponse(Gp,PID(Kcu,0,0),t)

Gc = PID(0.8*Kcu,0,Pu/8)
stepResponse(Gp,Gc,t)

tauC = 2
Kc = (2*zeta*tau)/(tauC + td)/Kp
tauI = 2*zeta*tau
tauD = tau/(2*zeta)

GcIMC = PID(Kc,0,tauD)
t = np.linspace(0,15,1000)
stepResponse(Gp,GcIMC,t)

fc = 0.157
fn = 1.12
K = 1.04
zetac = 1.71
tdelay = 1.01

tauc = 1/(2*np.pi*fc)
taun = 1/(2*np.pi*fn)

print(tauc,taun)

num,den = control.pade(tdelay,3)

GcPhys = K*control.tf([tauc,1],[taun**2,2*zetac*taun,1])
stepResponse(Gp,GcPhys,t)

u,t = control.step(Gc,t)
uPhys,t = control.step(GcPhys,t)
plt.plot(t,u,t,uPhys)
plt.xlabel('Time [sec]')
plt.ylabel('%SNA')
plt.title('Step Responses for ZN tuning and Physiological controllers')
plt.legend(['ZN Tuned PD Control','Measured Physiological Neural Arc']);

control.bode(GcPhys);
control.bode(Gc);

Gc = PID(0.12*Kcu,0,Pu/6)
Gcf = Gc*(control.tf([1],[1*taun,1]))**2
uf,t = control.step(Gcf,t)

uPhys,t = control.step(GcPhys,t)

plt.plot(t,uf,t,uPhys)
plt.legend(['Modified PD Controller','Neural Arc']);

stepResponse(Gp,Gcf,t)
stepResponse(Gp,GcPhys,t)
plt.legend(['Modified PD Control','Observed Neural Arc']);



