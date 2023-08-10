#%    _______
#%           |  q_in (manipulated)
#%           v
#%        -------
#%       |       |  Area = A1
#%       |       |
#%        -------
#%    ----   |
#    q_d  |  |  q1 = C1*sqrt(h1)
#%        v  v
#%        -------
#%       |       |  Area = A2
#%       |       |
#%        -------
#%           |
#%            -----> q2 = C2*sqrt(h2)
#%
# % The model equations are
# %
#%   dh(1)/dt = (qin(t) - C1*sqrt(h(1)))/A1
#%   dh(2)/dt = (qd(t) + C1*sqrt(h(1)) - C2*sqrt(h(2)))/A2
#%
#% q_d is an unmeasured inflow to tank 2. The control objective is to
#% maintain a desired level in tank 2 and thereby a steady outflow q2 by
#% adjusting the manipulable variable q_in.
#%
#% The right hand side of the differential equations are given in the form a
#% function of time (t), a vector of tank levels (h), and functions of time
#% that return values for qin(t) and qd(t)

# import imporant packages
import math 
import numpy
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pylab
from operator import add 
from scipy.interpolate import interp1d
get_ipython().magic('matplotlib inline')


# Model Parameters

A1 = 2.0;
C1 = 1.2;
A2 = 3.0;
C2 = 1.0;
h1store = [0];
h2store = [0];
Hmatrix = np.matrix([h1store,h2store])
qin = []
qd = []
# Model equations

def dHtankone(t,h1,qin,qd):
    dh = (qin - C1*np.real(math.sqrt(h1)))/A1
    return dh
def dHtanktwo(t,h1,h2,qin,qd):
    dhtwo = (qd + C1*np.real(math.sqrt(h1)) - C2*np.real(math.sqrt(h2)))/A2
    return dhtwo
dHtanktwo(10,10,10,10,10);

# nominal inlet flows
qin_nominal = 1.0;
qd_nominal = 0.0;

def qinfunction(t):
    qin = qin_nominal
    return qin
def qdfunction(t):
    qd = qd_nominal
    return qd
def height(M,t):
    h1 = M[0];
    h2 = M[1];
    qin = qinfunction(t);
    qd = qdfunction(t);
    X1 = dHtankone(t,h1,qin,qd)
    X2 = dHtanktwo(t,h1,h2,qin,qd)
    dH = [X1,X2]
    return dH
#start = [0,0,0,1]

#dH = height(start,0)
tf = 150;
h1_initial = 0;
h2_initial = 0;
#h1.append(h1_initial);
#h2.append(h2_initial);
#h1store.append(dH[0]);
#h2store.append(dH[1]);
#Hmatrix = np.matrix([h1store,h2store])

time = numpy.linspace(0.0, tf, 1000)
start = [0,0]

f = odeint(height, start, time)
print f
h_ss = f[-1,:]

fr, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(time,f[:,0],'b')
axarr[0].plot(time,f[:,1],'r')
axarr[0].set_title('Tank Levels')
axarr[0].axhline(y = h_ss[0], xmin = 0, xmax = 1, color='b',ls='--')
axarr[0].axhline(y = h_ss[1], xmin = 0, xmax = 1, color= 'r',ls='--')
axarr[0].legend(['Tank Level 1','Tank Level 2'],'best')
axarr[0].set_ylim([0,1.2])
axarr[0].set_ylabel('Level [meters]')
axarr[0].set_xlabel('Time[min]')
axarr[1].axhline(y = qinfunction(1), xmin = 0, xmax = 1, color='b')
axarr[1].axhline(y = qdfunction(1), xmin = 0, xmax = 1, color='r')
axarr[1].set_ylim([-.2,1.2])
axarr[1].legend(['Manipulated Inflow','Disturbance'],'best')
axarr[1].set_ylabel('Flow (cubic metters/min)')
axarr[1].set_xlabel('Time[min]')



# Establish a Reference function
def href(t):
    hreff = h_ss[1] + min(.1*t, .5) - .7*(t>=120)
    return hreff
# Redefine Qd
def qdfunction(t):
    qd = .3*((t>= 50) & (t<70));
    return qd
# Turn Time into Discreet Time
dt = 2.0;
tf = 200;
time_discrete = np.linspace(0,200,101)

h = h_ss
I = 0;
k = 0;
yrefstore = [0]
p1 = [h_ss[1]]
#p2 = [h_ss[0]]
#p3 = [h_ss[1]]
qinf = [1]

while k < len(time_discrete) - 1:
    # Our Measured Value
    y = h[1] - h_ss[1];
    # The PI Controller
    yref = href(time_discrete[k]) - h_ss[1];
    yerr = yref - y;
    P = .6*yerr;
    I = I + .1*dt*yerr;
    u = P + I;
    # Actuator 
    qin_last = qinfunction(time_discrete[k])
    qin_delta = (qin_nominal + u) - qin_last;
    def qinfunction(ts):
        qin = qin_last + (ts > time_discrete[k])*qin_delta;
        return qin
    tiwtp = qinfunction(time_discrete[k+1])
    qinf.append(tiwtp)
    # Process Simulation
    M = [h[0],h[1]]
    timeinter = [time_discrete[k],time_discrete[k+1]]
    Hsim = odeint(height, M, [time_discrete[k], time_discrete[k+1]])
    h = Hsim[-1,:]
    k = k+1
    # Visualization
    p1.append(y+h_ss[1])
    #p2.append(Hsim[-1,0])

f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(time_discrete, p1,'b-o',ms=4.0)
hrf = [href(0)]
time = numpy.linspace(0,200,201)
for t in range(1,201):
    hrf.append(href(t))
axarr[0].plot(time, hrf,'r--')
axarr[0].set_title('Feedback Control of Tank 2 Level')
axarr[0].set_ylabel('Level [meters]')

qdf = [qdfunction(0)]
for t in range(1,201):
    qdf.append(qdfunction(t))
    
axarr[1].plot(time, qdf,'r')
axarr[1].plot(time_discrete,qinf,'b',drawstyle='steps')
axarr[1].set_ylabel('Flow')
axarr[1].set_xlabel('Time[min]')
plt.ylim([0,2])

A = np.array([[-C1/(2*A1*math.sqrt(h_ss[0])), 0],[C1/(2*A2*math.sqrt(h_ss[0])), -C2/(2*A2*math.sqrt(h_ss[1]))]])
B = np.array([1/A1, 0])
E = np.array([0, 1/A2])
C = [0, 1]
D = [0]
F = [0]

# Now compare the response of the nonlinear model and the linear
# approximation for typical inputs. A time grid is set up, then used to for
# a nonlinear simulation starting at the nominal steady state and with some
# time-varying inputs. The simulation is then repeated for the nonlinear
# model after constructing the initial condition and inputs in terms of the
# deviations from the nominal steady state.

# simulation horizon
tf = 80
t = numpy.linspace(0,tf,1601)

# nonlinear simulation
def qinfunction(t):
    qin = qin_nominal + 0.2*(t>=10) - 0.4*(t>=40)
    return qin
h = odeint(height, h_ss, t)

# linear simulation
def ufunction(t):
    return qinfunction(t) - qin_nominal
def dfunction(t):
    return qdfunction(t) - qd_nominal

x_initial =  np.array([0,0])
def linfunction(x,t):
    return np.dot(A,x) + B*ufunction(t) + E*dfunction(t)

x = odeint(linfunction, x_initial, t)

#np.dot(A,x_initial) + np.dot(B,ufunction(15)) + np.dot(E, dfunction(15))

#%%
#% The simulation results are displayed by converting the linear deviation
# variables back to values corresponding to the actual process variables.
#% This allows for a better comparison of the nonlinear and linear models.

f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(t,h[:,0],'b')
axarr[0].plot(t,h[:,1],'r')
axarr[0].plot(t,(x+h_ss)[:,0],'b--')
axarr[0].plot(t,(x+h_ss)[:,1],'r--')
axarr[0].legend(['Tank Level 1','Tank Level 2'],'best')
axarr[0].set_ylabel('Level [meters]')
axarr[0].set_title('Response: Nonlinear (solid) versus Linear (dashed) models')
axarr[1].plot(t,qinfunction(t))
axarr[1].plot(t,qdfunction(t),'r')
axarr[1].set_ylabel('cubic meters/min')
axarr[1].set_xlabel('Time[min]')
axarr[1].legend(['qin manipulated input','qd disturbance'],'best')
plt.ylim([0,1.4])

#import expm function
from scipy.linalg import expm, inv

#sample time
dt=2.0;
A = np.matrix([[-C1/(2*A1*math.sqrt(h_ss[0])), 0],[C1/(2*A2*math.sqrt(h_ss[0])), -C2/(2*A2*math.sqrt(h_ss[1]))]])
B = np.array([[1/A1], [0]])
E = np.array([[0], [1/A2]])

#discrete-time model 
Ad=expm(A*dt);
Bd=(expm(A*dt)-numpy.matrix(numpy.identity(2)))*inv(A)*B;
Ed=(expm(A*dt)-numpy.matrix(numpy.identity(2)))*inv(A)*E;
Ad1=Ad[0]
Ad2=Ad[1]
Ad=np.matrix([Ad1, Ad2])

    

#%%
#% Next we construct simulations using both the discrete-time and
#% continuous-time models.

xi1 = [0];
xi2 = [0];
x_initial = np.matrix([xi1,xi2])

# discrete time simulation
td = numpy.linspace(0,tf,tf/dt+1);
xd = x_initial;

for k in range (1,len(td-1)):
    xd = np.c_[xd,np.dot(Ad,xd[:,k-1])+np.dot(Bd,ufunction(td[k-1]))+np.dot(Ed,dfunction(td[k-1]))]

# flip matrix to row vector
xd= xd.T

#continues-time model taken from Demo 3

f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(t,x[:,0],'b')
axarr[0].plot(t,x[:,1],'r')
axarr[0].plot(td,xd[:,0],'bo',ms=5.0)
axarr[0].plot(td,xd[:,1],'ro',ms=5.0)
axarr[0].set_ylabel('Deviation from SS')
axarr[0].set_title('System Response')
axarr[0].legend(['Tank Level 1','Tank Level 2'],'lower left')

axarr[1].plot(t,ufunction(t),'b')
axarr[1].plot(t,dfunction(t),'r')
axarr[1].plot(td,ufunction(td),'o')
axarr[1].plot(td,dfunction(td),'ro')
axarr[1].set_ylabel('Deviation from SS')
axarr[1].set_xlabel('Time[min]')
axarr[1].legend(['qin manipulated input','qd disturbance'],'lower left')

plt.ylim([-.3,.3])


#% prediction and control horizons
dt = 2.0;
tprediction = 50;
tcontrol = 8;

tpred = np.linspace(0,tprediction+tcontrol,((tprediction+tcontrol)/dt)+1);

#% unit step input
def ufunction(t):
    upred=1.0;
    return upred

#% compute unit step response
x1=[0]
x2=[0]
x = np.matrix([x1,x2]);

#cant make ystep an empty matrix leads to extra point on graph
ystep=[0];

for k in range(0,len(tpred)):
    ystep = np.c_[ystep,np.dot(C,x)+np.dot(D,ufunction(tpred[k]))]
    x= np.dot(Ad,x) + np.dot(Bd,ufunction(tpred[k]))
    
ystep = ystep.T[range(1,31)]
#weird range ^^ because of extra point from ystep

plt.plot(tpred,ystep,'bo',ms=4.0)
plt.ylabel('Deviation of Tank Level2 from SS [meters]')
plt.xlabel('Time[min]')
plt.title('Step Response')

## Establish Toeplitz Matrix

from scipy.linalg import toeplitz
# toeplitz function 
steps = int(tcontrol/dt);

ytoe = ystep

ytoe = np.array(ytoe)
ytoe = ytoe.tolist()
zerovector = np.zeros(steps+1)
S = toeplitz(ytoe,zerovector)

f, axarr = plt.subplots(2, sharex=True)

for k in range(0,len(tpred)-1):
    axarr[0].plot(tpred[k],href(tpred[k]),'ro')
plt.ylim(0,2)

#%%
#% A series of control moves du are computed by 'solving' the equation
#%
#%   yref = S*du
#%
#% where
#% 
#%   yref = href = h_ss
#%
#% and u is reconstructed from du by taking a cumulative sum. The quotes
#% around 'solving' are because there are more equations than unknowns, so
#% the best we can do (without further knowledge) is a regression.

def yreffunction(t):
    yref= href(t) - h_ss[1]
    return yref
yreffunction
yrefsave=np.zeros(len(S))
for k in range(0,len(S)):
    #yref= np.c_[yref,yreffunction(tpred[k])]
    #print yref
    yrefsave[k]= yreffunction(tpred[k])
    #yref = np.vstack((yref,yrefnew))

np.array(yrefsave)
#print len(yrefsave)
#print len(S)
S = np.reshape(S,(len(S),steps+1))
du= np.linalg.lstsq(S,yrefsave)[0]
#du = S\yref[tpred]
du=du.T
print du

## The Following Function Was supplied by Professor Jeff Kantor, University of Notre Dame
def interp0(x, xp, yp):
    """Zeroth order hold interpolation w/ same
    (base)   signature  as numpy.interp."""

    def func(x0):
        if x0 <= xp[0]:
            return yp[0]
        if x0 >= xp[-1]:
            return yp[-1]
        k = 0
        while x0 > xp[k]:
            k += 1
        return yp[k-1]

    return map(func, x)


## Convert Control Moves to a Continuous Control Policy
from scipy import interp
linearrange = np.linspace(0,tcontrol,tcontrol/dt+1)
linearrange = np.array(linearrange)
def ufunctionvec(t):
    return interp0(t,linearrange, np.cumsum(du))
def qinfunctionvec(t):
    X = (qin_nominal*np.ones(len(linearrange))).T + ufunction(linearrange)
    return X

R = h_ss[1] + np.dot(S,du)
axarr[0].plot(tpred, R,'bo',ms=4.0)
axarr[0].set_ylim(0,2)
#plt.plot()

## The Following Function Was supplied by Professor Jeff Kantor, University of Notre Dame
def interp0(x, xp, yp):
    """Zeroth order hold interpolation w/ same
    (base)   signature  as numpy.interp."""

    def func(x0):
        if x0 <= xp[0]:
            return yp[0]
        if x0 >= xp[-1]:
            return yp[-1]
        k = 0
        while x0 > xp[k]:
            k += 1
        return yp[k-1]

    return func(x)




def ufunction(t):
    return interp0(t,linearrange, np.cumsum(du))
def qinfunction(t):
    X = (qin_nominal) + ufunction(t)
    return X



t = np.linspace(0,tpred[-1],2901)
michael = np.ones(len(t))
#print len(michael)
for l in range(0,len(t)):
    michael[l] = qinfunction(t[l])
    
#print len(michael)
axarr[1].plot(t,michael,'b')
#print len(t) 
#print len(qinfunction(t))
#plt.plot(t,interp0(t,linearrange, np.cumsum(du)))

tvec = np.linspace(0,58,30)

h = odeint(height, h_ss, tvec)
           
axarr[0].plot(tpred,h[:,0])
axarr[0].plot(tpred,h[:,1],'r')
axarr[1].set_ylim([1,2.2])
axarr[0].set_ylabel('Level [meters]')
axarr[0].set_title('Feedback Control of Tank 2 Level')
axarr[1].set_ylabel('Flow')
axarr[1].set_xlabel('Time[min]')

dt = 2
t = np.linspace(0,200,101)
ypred = np.zeros(len(tpred))
h = h_ss
u = 0

def qinfunction(t):
    return qin_nominal
qin_min = 0.0
qin_max = 2.0
du_max = .5
du_min = -.5

def href(t):
    hreff = h_ss[1] + min(.1*t, .5) - .7*(t>=120)
    return hreff

hrefvector = np.zeros(len(t))
for i in range(0,len(t)):
    hrefvector[i] = href(t[i])
    
# build subplot    
fr, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(t,hrefvector,'r--',linewidth = 2.0)
axarr[1].plot(t,qdfunction(t),'r')
h = h_ss

for k in range(0,len(t)-1):
    leastsquare = np.zeros(len(tpred))
    y = h[1] - h_ss[1]
    ## Shift Prediction
    ypredsave = (ypred[-1])
    ypredremove = (ypred[0])
    ypred = ypred.tolist()
    ypred.remove(ypredremove)
    ypred.append(ypredsave)
    ydisturbance = np.add(y,-ypred[0])
    ypred = np.add(ypred,ydisturbance) 
    for j in range(0,len(tpred)):
        leastsquare[j] = yreffunction(t[k] + tpred[j]) - ypred[j]
    du = np.linalg.lstsq(S,leastsquare)[0][0]
    du = min(du, qin_max-qinfunction(t[k]))
    du = max(du, qin_min-qinfunction(t[k]))
    du = min(du, du_max)
    du = max(du, du_min)
    ypred = np.add(ypred,S[:,0]*du)
    qin_last = qinfunction(t[k+1])
    def qinfunction(ts):
        A = qin_last + duf*(ts > t[k])
        return A
    duf = du
    timevector = [t[k], t[k+1]]
    f = odeint(height,h,timevector)
    h = f[1,:]
    plt.hold(True)
    axarr[0].plot([t[k], t[k+1]],[f[0,1],f[1,1]], 'r')
    axarr[0].plot([t[k], t[k+1]],[f[0,0],f[1,0]], 'b')
    axarr[0].set_ylim([0,2])
   
    axarr[1].plot([t[k],t[k+1]], [qinfunction(t[k]), qinfunction(t[k+1])], 'b',drawstyle='steps')
    axarr[0].set_ylabel('Level [meters]')
    axarr[1].set_ylabel('Flow')
    axarr[1].set_xlabel('Time[min]')
    axarr[0].set_title('Feedback Control of Tank 2 Level')

from cvxopt import matrix
from cvxopt.modeling import variable, op, max, sum 
from scipy.optimize import minimize
from cvxopt import *
from cvxpy import *

# discrete simulation horizon 
dt = 2.0
t = np.linspace(0,200,101)

# Vector to predicted values of y
ypred = np.zeros(len(tpred))

# current tank heights
h = h_ss
u = 0

# Initialize control
def qinfunction(t):
    return qin_nominal

# plotting
hrefvector = np.zeros(len(t))
for i in range(0,len(t)):
    hrefvector[i] = href(t[i])
    
#build subplot
fr, axarr = plt.subplots(2, sharex=True)
    
axarr[0].plot(t,hrefvector,'r--')
axarr[1].plot(t,qdfunction(t),'r')

for k in range(0,len(t)-1):  #####
    # get current measurment
    y = h[1]-h_ss[1]
    # shift prediction to new time
    ypredsave = (ypred[-1])
    ypredremove = (ypred[0])
    ypred = ypred.tolist()
    ypred.remove(ypredremove)
    ypred.append(ypredsave)
    # compare measurment to prediction
    ydisturbance = np.add(y,-ypred[0])
    # measurment mismatch update
    ypred = np.add(ypred,ydisturbance) 
    
    # Compute control moves needed push ypred to yref
    dy = np.zeros(len(tpred))
    for j in range(0,len(tpred)):
        dy[j] = yreffunction(t[k] + tpred[j]) - ypred[j]

    # Create two scalar optimization variables.
    a = Variable()
    b = Variable()
    c = Variable()
    d = Variable()
    e = Variable()
    F = np.array([a,b,c,d,e])
    G = a
    H = a+b
    I = H+c
    J = I+d
    K = J+e

    constraints = [G + qinfunction(t[k]) <= qin_max,
                   H + qinfunction(t[k]) <= qin_max,
                   I + qinfunction(t[k]) <= qin_max,
                   J + qinfunction(t[k]) <= qin_max,
                   K + qinfunction(t[k]) <= qin_max,
                   G + qinfunction(t[k]) >= qin_min,
                   H + qinfunction(t[k]) >= qin_min,
                   I + qinfunction(t[k]) >= qin_min,
                   J + qinfunction(t[k]) >= qin_min,
                   K + qinfunction(t[k]) >= qin_min,
                   a <= du_max,
                   b <= du_max,
                   c <= du_max,
                   d <= du_max,
                   e <= du_max,
                   a >= du_min,
                   b >= du_min,
                   c >= du_min,
                   d >= du_min,
                   e >= du_min]
    
    # Form objective.
    def happy1(s):
        a = len(s)
        d = 0
        for k in range(0,a):
            d = d + abs(s[k])
        return d
        
    obj = Minimize(happy1(np.dot(S,F)-dy))

    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()
    
    # control update assuming we implement the first control move
    ypred = np.add(ypred,S[:,0]*a.value)
    # Actuator model
    qin_last = qinfunction(t[k+1])
    def qinfunction(ts):
        A = qin_last + duf*(ts > t[k])
        return A
    duf = a.value
    # process simulation
    timevector = [t[k], t[k+1]]
    f = odeint(height,h,timevector)
    h = f[1,:]
    
    
    # Visualization
    axarr[0].plot([t[k], t[k+1]], [f[0,1],f[1,1]], 'r-')
    axarr[0].plot([t[k], t[k+1]], [f[0,0], f[1,0]] ,'b' )
    axarr[0].set_title('Feedback Control of Tank 2 Level')
    axarr[0].set_ylabel('Level[meters]')
    axarr[1].set_ylabel('Flow')
    axarr[1].set_xlabel('Time [mins]')
    axarr[1].plot([t[k], t[k+1]], [qinfunction(t[k]),qinfunction(t[k+1])], 'b',linestyle='steps')
    # Plot right above is on flow plot in matlab
    


# maximum level for tank 1
x_max = 5
x_min = 0

# compute unit step response of tank 1
x = np.array([[0],[0]])
c = np.zeros(len(tpred))
for k in range(0,len(tpred)):
    c[k] = x[0]
    x = np.dot(Ad,x) + Bd
    

yzero = np.zeros(5)

R = toeplitz(c, yzero)

# vector to preicted values of y
xpred = np.zeros(len(tpred))
ypred = np.zeros(len(tpred))

# current tank heights
h = h_ss
u = 0

# min an max flowrates
def qinfunction(t):
    return qin_nominal 

fr, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(t, hrefvector, 'r--',linewidth=2.0)
axarr[1].plot(t,qdfunction(t), 'r')
bbb = np.zeros(len(t))

for k in range(0,len(t)-1):
    # get current measurment
    x = h[0] - h_ss[0]
    y = h[1] - h_ss[1]
    
    # shift prediction to new time
    ypredsave = (ypred[-1])
    ypredremove = (ypred[0])
    ypred = ypred.tolist()
    ypred.remove(ypredremove)
    ypred.append(ypredsave)
    # compare measurment to prediction
    ydisturbance = np.add(y,-ypred[0])
    # measurment mismatch update
    ypred = np.add(ypred,ydisturbance) 
    
    # shift prediction to new time
    xpredsave = (xpred[-1])
    xpredremove = (xpred[0])
    xpred = xpred.tolist()
    xpred.remove(xpredremove)
    xpred.append(xpredsave)
    # compare measurment to prediction
    xdisturbance = np.add(x,-xpred[0])
    # measurment mismatch update
    xpred = np.add(xpred,xdisturbance) 
    
    dy = np.zeros(len(tpred))
    for j in range(0,len(tpred)):
        dy[j] = yreffunction(t[k] + tpred[j]) - ypred[j]
           
                
        
        
        
    # Create two scalar optimization variables.
    a = Variable()
    b = Variable()
    c = Variable()
    d = Variable()
    e = Variable()
    F = np.array([a,b,c,d,e])
    G = a
    H = a+b
    I = H+c
    J = I+d
    K = J+e
    YY = np.add(h_ss[0],xpred)
    XX = np.dot(R,F)
    GG = YY + XX
    l = GG[0]
    m = GG[1]
    n = GG[2]
    o = GG[3]
    p = GG[4]

    constraints = [G + qinfunction(t[k]) <= qin_max,
                   H + qinfunction(t[k]) <= qin_max,
                   I + qinfunction(t[k]) <= qin_max,
                   J + qinfunction(t[k]) <= qin_max,
                   K + qinfunction(t[k]) <= qin_max,
                   G + qinfunction(t[k]) >= qin_min,
                   H + qinfunction(t[k]) >= qin_min,
                   I + qinfunction(t[k]) >= qin_min,
                   J + qinfunction(t[k]) >= qin_min,
                   K + qinfunction(t[k]) >= qin_min,
                   a <= du_max,
                   b <= du_max,
                   c <= du_max,
                   d <= du_max,
                   e <= du_max,
                   a >= du_min,
                   b >= du_min,
                   c >= du_min,
                   d >= du_min,
                   e >= du_min,
                   l <= x_max,
                   m <= x_max,
                   n <= x_max,
                   o <= x_max,
                   p <= x_max,
                   l >= x_min,
                   m >= x_min,
                   n >= x_min,
                   o >= x_min,
                   p >= x_min]

    # Form objective.
    def happy1(s):
        a = len(s)
        d = 0
        for k in range(0,a):
            d = d + abs(s[k])
        return d
        
    obj = Minimize(happy1(np.dot(S,F)-dy))

    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()
    
    cc = a.value
    bbb[k] = b.value
    if a.value == None:
        cc = 0
    # control update assuming we implement the first control move

    
    ypred = np.add(ypred,S[:,0]*cc)
    xpred = np.add(xpred,R[:,0]*cc)
    
    
    # actuator model 
    qin_last = qinfunction(t[k+1])
    def qinfunction(ts):
        A = qin_last + duf*(ts > t[k])
        return A
    duf = cc
    # process simulation
    timevector = [t[k], t[k+1]]
    f = odeint(height,h,timevector)
    h = f[1,:]
    
    axarr[0].plot([t[k], t[k+1]], [f[0,1],f[1,1]], 'r')
    axarr[0].plot([t[k], t[k+1]], [f[0,0], f[1,0]] ,'b' )
    axarr[0].set_title('Feedback Control of Tank 2 Level')
    axarr[0].set_ylabel('Level[meters]')
    axarr[1].set_ylabel('Inlet Flow')
    axarr[1].set_xlabel('Time [mins]')
    axarr[1].plot([t[k], t[k+1]], [qinfunction(t[k]),qinfunction(t[k+1])], 'b',linestyle='steps')# maximum level for tank 1



