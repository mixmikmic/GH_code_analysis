get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constant parameter values as used in Figure 8 of the paper
sn=0.073      # cells/microliter
dn=0.040      # 1/day
de=0.06       # 1/day
dc=0.2        # 1/day
kn=0.001      # 1/day
eta=100       # cells/microliter
alpha_n=0.41  # unitless
alpha_e=0.2   # 1/day
rc=0.03       # 1/day
gamma_e=0.005 # microliter/day*cell
gamma_c=0.005 # microliter/day*cell
Cmax=300000   # cells/microliter

# non-dimensionalized coefficents
zeta1=kn/dn
zeta2=(gamma_e*eta)/dn
zeta3=(alpha_n*kn*sn*gamma_c)/(dn**3.0)
zeta4=alpha_e/dn
zeta5=de/dn
zeta6=rc/dn
zeta7=(gamma_e*Cmax)/dn
zeta8=dc/dn

# Analysis in the absence of T cell therapy, u=0
c = np.linspace(0.0000001,60,1000000)

dCdt=-c*(zeta8-zeta6*np.log(zeta7)+zeta6*np.log(c)+((zeta3*c*(c+zeta2))/((c+zeta2+zeta1*c)*((c+zeta2)*(c+zeta5)-zeta4))))

plt.plot(c,dCdt)
plt.xlabel('C')
plt.ylabel('dC/dt')
plt.title('Nondimensionalized dC/dt vs C')
plt.grid()
CancerSteadyState = 47/(gamma_e/dn)
print('Cancer Steady State (cells/microliter)')
print(CancerSteadyState)

# Analysis with a constant, non-zero value of u(t).
c = np.linspace(0.0000001,40,100000)
# The unscaled values of u are the actual infusion rates before they are nondimensionalized to be fed into the equation
u1_unscaled=4.4    # cells/microliter*day
u2_unscaled=4.6    # cells/microliter*day
u3_unscaled=4.8    # cells/microliter*day
# These are the nondimensionalized u values
u1 = u1_unscaled*gamma_c/dn**2
u2 = u2_unscaled*gamma_c/dn**2;
u3 = u3_unscaled*gamma_c/dn**2;
# Each nondimensionalized u value is fed into the nondimensionlized form of dC/dt
dCdtU1=-c*(zeta8-zeta6*np.log(zeta7)+zeta6*np.log(c)+(((c+zeta2)*(c*u1*zeta1+u1*zeta2+c*(u1+zeta3)))/((c+zeta2+zeta1*c)*((c+zeta2)*(c+zeta5)-zeta4))))
dCdtU2=-c*(zeta8-zeta6*np.log(zeta7)+zeta6*np.log(c)+(((c+zeta2)*(c*u2*zeta1+u2*zeta2+c*(u2+zeta3)))/((c+zeta2+zeta1*c)*((c+zeta2)*(c+zeta5)-zeta4))))
dCdtU3=-c*(zeta8-zeta6*np.log(zeta7)+zeta6*np.log(c)+(((c+zeta2)*(c*u3*zeta1+u3*zeta2+c*(u3+zeta3)))/((c+zeta2+zeta1*c)*((c+zeta2)*(c+zeta5)-zeta4))))

# Plotting up the nondimensionalized results of dC/dt with constant u values
plt.plot(c,dCdtU1, label = 'u = 4.4 cells/microliter*day')
plt.plot(c,dCdtU2, label = 'u = 4.6 cells/microliter*day')
plt.plot(c,dCdtU3, label = 'u = 4.8 cells/microliter*day')
plt.xlabel('C')
plt.ylabel('dC/dt')
plt.title('dC/dt vs C [Constant u(t)]')
plt.grid()
plt.legend(loc='lower left')

# plotting on a smaller interval to see the steady state at 0
c = np.linspace(0,5,10000)
dCdtU1_=-c*(zeta8-zeta6*np.log(zeta7)+zeta6*np.log(c)+(((c+zeta2)*(c*u1*zeta1+u1*zeta2+c*(u1+zeta3)))/((c+zeta2+zeta1*c)*((c+zeta2)*(c+zeta5)-zeta4))))
dCdtU2_=-c*(zeta8-zeta6*np.log(zeta7)+zeta6*np.log(c)+(((c+zeta2)*(c*u2*zeta1+u2*zeta2+c*(u2+zeta3)))/((c+zeta2+zeta1*c)*((c+zeta2)*(c+zeta5)-zeta4))))
dCdtU3_=-c*(zeta8-zeta6*np.log(zeta7)+zeta6*np.log(c)+(((c+zeta2)*(c*u3*zeta1+u3*zeta2+c*(u3+zeta3)))/((c+zeta2+zeta1*c)*((c+zeta2)*(c+zeta5)-zeta4))))
plt.plot(c,dCdtU1_, label = 'u = 4.4 cells/microliter*day')
plt.plot(c,dCdtU2_, label = 'u = 4.6 cells/microliter*day')
plt.plot(c,dCdtU3_, label = 'u = 4.8 cells/microliter*day')
plt.xlabel('C')
plt.ylabel('dC/dt')
plt.title('dC/dt vs C [Constant u(t)]')
plt.grid()
plt.legend(loc='upper right')

# Plotting up the values of the naive T cells (u(t)=constant)
# It is important to note that this population is unaffected by u(t).
c = np.linspace(0.000001,100,100000)
Tn = (c+zeta2)/(c+c*zeta1+zeta2)
plt.plot(c,Tn)
plt.xlabel('C')
plt.ylabel('Naive T cells')
plt.title('Naive T cells vs. Cancer cells')
plt.grid()

# Calculating populations of effector T cells (u(t)=constant)
c = np.linspace(0.000001,60,100000)
Te1 = ((c+zeta2)*(c*u1*zeta1+u1*zeta2+c*u1+c*zeta3))/((c+c*zeta1+zeta2)*(zeta2*c+zeta2*zeta5+c*c-c*zeta4+c*zeta5))
Te2 = ((c+zeta2)*(c*u2*zeta1+u2*zeta2+c*u2+c*zeta3))/((c+c*zeta1+zeta2)*(zeta2*c+zeta2*zeta5+c*c-c*zeta4+c*zeta5))
Te3 = ((c+zeta2)*(c*u3*zeta1+u3*zeta2+c*u3+c*zeta3))/((c+c*zeta1+zeta2)*(zeta2*c+zeta2*zeta5+c*c-c*zeta4+c*zeta5))

# Plotting up the effector T cell populations (u(t)=constant)
plt.plot(c,Te1, label = 'u=4.4 cells/microliter*day')
plt.plot(c,Te2, label = 'u=4.6 cells/microliter*day')
plt.plot(c,Te3, label = 'u=4.8 cells/microliter*day')
plt.xlabel('C')
plt.ylabel('Effector T cells')
plt.title('Effector T cells vs. Cancer cells [Constant u(t)]')
plt.grid()
plt.legend()

# Subplots of each cell type population - for comparison
# These are all nondimensionalized graphs

# Cancer cell population
plt.subplot(3,1,1)
c = np.linspace(0.000001,100,100000)
plt.plot(c,dCdtU1, label = 'u=4.4 cells/microliter*day')
plt.plot(c,dCdtU2, label = 'u=4.6 cells/microliter*day')
plt.plot(c,dCdtU3, label = 'u=4.8 cells/microliter*day')
plt.xlabel('C')
plt.ylabel('dC/dt')
plt.title('dC/dt vs C [Constant u(t)]')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Naive T cell population
plt.subplot(3,1,2)
Tn = (c+zeta2)/(c+c*zeta1+zeta2)
plt.plot(c,Tn)
plt.xlabel('C')
plt.ylabel('Naive T cells')
plt.title('Naive T cells vs. Cancer cells')
plt.grid()

# Effector T cell population
plt.subplot(3,1,3)
plt.plot(c,Te1, label = 'u=4.4 cells/microliter*day')
plt.plot(c,Te2, label = 'u=4.6 cells/microliter*day')
plt.plot(c,Te3, label = 'u=4.8 cells/microliter*day')
plt.xlabel('C')
plt.ylabel('Effector T cells')
plt.title('Effector T cells vs. Cancer cells [Constant u(t)]')
plt.grid()

plt.subplots_adjust(hspace=.8, top=1.5)

# Integrating the system of three differential equations to see their interactions for a constant u(t)
# Integrating the differential equations with respect to time

# defining our constant u(t) value
def u(t):
    return 4.6 # cells/microliter*day

# defining the system of differential equations - nondimensionalized
def deriv(X,t):
    Tn,Te,C = X
    dTn = 1-Tn-zeta1*Tn*(C/(C+zeta2))
    dTe = zeta3*Tn*(C/(C+zeta2))+zeta4*Te*(C/(C+zeta2))-zeta5*Te-C*Te+u(t)*gamma_c/dn**2
    dC = zeta6*C*(np.log(zeta7)-np.log(C))-zeta8*C-C*Te
    return [dTn,dTe,dC]

# Initial conditions - nondimensionalized
Tn0 = 1510*(dn/sn)
Te0 = 20*(gamma_c/dn)
# The paper gave 10,000 as the initial condition for cancer; we chose 400 since the steady state is ~376
C0 = 400*(gamma_e/dn)


IC = [Tn0, Te0, C0]

t = np.linspace(0,700*dn,1000)    # dimensionless time

X = odeint(deriv, IC, t)
Tn=X[:,0]
Te=X[:,1]
C=X[:,2]


# visualization
# time is re-dimensionalized here, as is C, Te, and Tn
t_dim = t/dn               # days
Tn_dim = Tn/(dn/sn)        # cells/microliter
Te_dim = Te/(gamma_c/dn)   # cells/microliter
C_dim = C/(gamma_e/dn)     # cells/microliter
plt.semilogy(t_dim,Tn_dim,label = 'Naive T Cells')
plt.semilogy(t_dim,Te_dim,label = 'Effector T Cells')
plt.semilogy(t_dim,C_dim,label = 'Cancer Cells')

plt.legend(loc='lower left')
plt.xlabel('Time (days)')
plt.ylabel('Cell Concentration (cells/microliter)')
plt.title('Cell Concentration over Time')
plt.grid()

# Plotting up cancer cell populations with several different u(t) values
t = np.linspace(0,2000*dn,1000)
t_dim = t/dn
def deriv(X,t):
    Tn,Te,C = X
    dTn = 1-Tn-zeta1*Tn*(C/(C+zeta2))
    dTe = zeta3*Tn*(C/(C+zeta2))+zeta4*Te*(C/(C+zeta2))-zeta5*Te-C*Te+u*gamma_c/dn**2
    dC = zeta6*C*(np.log(zeta7)-np.log(C))-zeta8*C-C*Te
    return [dTn,dTe,dC]

# defining a list of constant u values (including our value of 4.6 used previously)
uList = np.linspace(3.7,4.6,10)
for u in uList:
    X = odeint(deriv, IC, t)
    C = X[:,2]
    C_dim = C/(gamma_e/dn) 
    plt.semilogy(t_dim,C_dim)
plt.legend(uList,bbox_to_anchor=(1.3, 1))
plt.title('Cancer Cell Population with Various Constant Values of u(t)')
plt.xlabel('Time (days)')
plt.ylabel('Cancer Cell Concentration (cells/microliter)')

# Plotting up on an even smaller range to find the critical u value
# Plotting up cancer cell populations with several different u values
t = np.linspace(0,3000*dn,1000)
t_dim = t/dn
uList = np.linspace(3.88,3.9,3)
for u in uList:
    X = odeint(deriv, IC, t)
    C = X[:,2]
    C_dim = C/(gamma_e/dn) 
    plt.semilogy(t_dim,C_dim)
plt.legend(uList,bbox_to_anchor=(1.3, 1))
plt.title('Cancer Cell Population with Varrying Constant Values of u')
plt.xlabel('Time (days)')
plt.ylabel('Concentration (cells/microliter)')

# Implementing a dosing function, instead of a constant infusion of effector T cells
# parameter values
td = (1/24)    # administration time, days
tdose = 1      # time between doses, days
Udose = 5      # cells/microliter

t = np.linspace(0,400*dn,100000)

# defining a u(t) dosing regimen
def u(t):
    if (t/dn) % tdose <= td:
        return Udose/td
    else:
        return 0

# defining the same system as before, but now with our step funciton for u(t)
def deriv(X,t):
    Tn,Te,C = X
    dTn = 1-Tn-zeta1*Tn*(C/(C+zeta2))
    dTe = zeta3*Tn*(C/(C+zeta2))+zeta4*Te*(C/(C+zeta2))-zeta5*Te-C*Te+u(t)*gamma_c/dn**2
    dC = zeta6*C*(np.log(zeta7)-np.log(C))-zeta8*C-C*Te
    return [dTn,dTe,dC]

# same integration, but new system containing step funciton for u(t)
X = odeint(deriv, IC, t, tcrit=t)
Tn=X[:,0]
Te=X[:,1]
C=X[:,2]

# re-dimensionalizing the values
Tn_dim = Tn/(dn/sn)        # cells/microliter
Te_dim = Te/(gamma_c/dn)   # cells/microliter
C_dim = C/(gamma_e/dn)     # cells/microliter

# plotting it up
plt.semilogy(t/dn,Tn_dim,label = 'Naive T Cells')
plt.semilogy(t/dn,Te_dim,label = 'Effector T Cells')
plt.semilogy(t/dn,C_dim,label = 'Cancer Cells')
plt.legend(loc='lower left')
plt.xlabel('Time (days)')
plt.ylabel('Cell Concentration (cells/microliter)')
plt.title('Cell Concentration over Time')
plt.grid()

# Plotting up Te population on a shorter range to illustate dosing effects')
plt.plot(t/dn,Te_dim)
plt.axis([150,250,0,100])
plt.xlabel('Time (days)')
plt.ylabel('Cell Concentration (cells/microliter)')
plt.title('Effector T Cell Concentration over Time')
plt.grid()

# Dosing schedule - shown over a 10 day period to be seen clearly
y = [u(t) for t in t]
plt.plot(t/dn,y)
plt.title('Dosing Schedule')
plt.xlabel('Time (days)')
plt.ylabel('Administration Rate (cells/microliter*day)')
plt.axis([0,10,0,130])

t = np.linspace(0,400*dn,100000)
t_dim = t/dn

# Defining new values of Udose
Udose1 = 3   # cells/microliter
Udose2 = 4   # cells/microliter
Udose3 = 5   # cells/microliter
Udose4 = 6   # cells/microliter

# Defining a dose regemin for each Udose
def u1(t):
    if (t/dn) % tdose <= td:
        return Udose1/td
    else:
        return 0

def u2(t):
    if (t/dn) % tdose <= td:
        return Udose2/td
    else:
        return 0

def u3(t):
    if (t/dn) % tdose <= td:
        return Udose3/td
    else:
        return 0

def u4(t):
    if (t/dn) % tdose <= td:
        return Udose4/td
    else:
        return 0

# Defining a new system of differential equations for each u(t)
def deriv1(X1,t):
    Tn,Te,C = X1
    dTn = 1-Tn-zeta1*Tn*(C/(C+zeta2))
    dTe = zeta3*Tn*(C/(C+zeta2))+zeta4*Te*(C/(C+zeta2))-zeta5*Te-C*Te+u1(t)*gamma_c/dn**2
    dC = zeta6*C*(np.log(zeta7)-np.log(C))-zeta8*C-C*Te
    return [dTn,dTe,dC]

def deriv2(X2,t):
    Tn,Te,C = X2
    dTn = 1-Tn-zeta1*Tn*(C/(C+zeta2))
    dTe = zeta3*Tn*(C/(C+zeta2))+zeta4*Te*(C/(C+zeta2))-zeta5*Te-C*Te+u2(t)*gamma_c/dn**2
    dC = zeta6*C*(np.log(zeta7)-np.log(C))-zeta8*C-C*Te
    return [dTn,dTe,dC]

def deriv3(X3,t):
    Tn,Te,C = X3
    dTn = 1-Tn-zeta1*Tn*(C/(C+zeta2))
    dTe = zeta3*Tn*(C/(C+zeta2))+zeta4*Te*(C/(C+zeta2))-zeta5*Te-C*Te+u3(t)*gamma_c/dn**2
    dC = zeta6*C*(np.log(zeta7)-np.log(C))-zeta8*C-C*Te
    return [dTn,dTe,dC]

def deriv4(X4,t):
    Tn,Te,C = X4
    dTn = 1-Tn-zeta1*Tn*(C/(C+zeta2))
    dTe = zeta3*Tn*(C/(C+zeta2))+zeta4*Te*(C/(C+zeta2))-zeta5*Te-C*Te+u4(t)*gamma_c/dn**2
    dC = zeta6*C*(np.log(zeta7)-np.log(C))-zeta8*C-C*Te
    return [dTn,dTe,dC]

# Solving each system and storing the solutions
X1 = odeint(deriv1, IC, t, tcrit=t)
C1=X1[:,2]

X2 = odeint(deriv2, IC, t, tcrit=t)
C2=X2[:,2]

X3 = odeint(deriv3, IC, t, tcrit=t)
C3=X3[:,2]

X4 = odeint(deriv4, IC, t, tcrit=t)
C4=X4[:,2]

# Re-dimensionalizing the cell populations
C_dim1 = C1/(gamma_e/dn)     # cells/microliter
C_dim2 = C2/(gamma_e/dn)     # cells/microliter
C_dim3 = C3/(gamma_e/dn)     # cells/microliter
C_dim4 = C4/(gamma_e/dn)     # cells/microliter

# Plotting up the respective cancer cell populations
plt.semilogy(t/dn,C_dim1,label = '3 cells/microliter')
plt.semilogy(t/dn,C_dim2,label = '4 cells/microliter')
plt.semilogy(t/dn,C_dim3,label = '5 cells/microliter')
plt.semilogy(t/dn,C_dim4,label = '6 cells/microliter')
plt.legend(bbox_to_anchor=(1.5, 1))
plt.xlabel('Time (days)')
plt.ylabel('Cell Concentration (cells/microliter)')
plt.title('Cancer Cell Concentration over Time')



