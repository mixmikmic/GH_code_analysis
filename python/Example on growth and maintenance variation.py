import numpy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
get_ipython().magic('matplotlib inline')
alpha, beta, gamma = 0.1, 0.1,1.8
Km_mu, Km_theta = 0.0004, 0.00005
Cp_star, Kp = 1.95, 2.5  
mumax, thetamax = 0.12, 0.1

def response_fun(C):
    
    Cx, Cs, Cg, Ce = C[0],C[1],C[2],C[3]   #Same component order as in the ferm-eq
    Cp = Ce                                #Ethanol is Cp 
    
    mu1=mumax*Cs/(Km_mu+Cs)*(1-Cp/Cp_star)
    theta=thetamax*Cs/(Km_theta+Cs)*(1+Cp/Kp)**(-1)
    
    if mu1 < 0.0001:   #Negative values of mu set to zero
        mu=0 
    else:
        mu=mu1
                 
    S = numpy.matrix([[-1,1+alpha,1,0,0],
                      [ 0,0,-1,1,1.5],
                      [  0,beta,0,-1/3,0],
                      [ 0,-gamma,-1/3,0,1],
                      [ 0,1,0,0,0]])

    K = numpy.matrix([0, 0, 0, theta, mu]).T
    r = numpy.linalg.solve(S, K) 
        
    
    return [r[1,0],   # important to see the order of the rates (X,S,G,E)
           -r[0,0],   # glucose, note the negative since the flux model calculated (-rs)
            r[3,0],   # glycerol
            r[4,0],   # ethanol   
            theta]

rate1=response_fun([12/24.6,60/30,0,20/46])
rate1

Yse1 = rate1[3]/-rate1[1]
Yse1

rate2=response_fun([20/24.6,15/30,0,100/46])
rate2

Yse2 = rate2[3]/-rate2[1]
Yse2

Cp_curve=numpy.linspace(0,3,100)
mu_curve=mumax*(1-Cp_curve/Cp_star)
theta_curve=thetamax*(1+Cp_curve/Kp)**(-1)

#% matplotlib notebook   
plt.figure(1)
plt.plot(Cp_curve, mu_curve, color='blue', label='mu')
plt.plot(Cp_curve, theta_curve, color='red', label='theta')
plt.ylim([0,0.14])  #y limits on graph
plt.legend(loc='best')
plt.ylabel('rate') 
plt.xlabel('Concentration CP (cmol\L)') 
plt.show()

def ferm_eq(C,t):
    Cx, Cs, Cg, Ce = C[0],C[1],C[2],C[3]  # order of components very important 
    r=response_fun([Cx, Cs, Cg, Ce])      # this is the 'fetching' from the response function
    
    return [r[0]*Cx,
            r[1]*Cx,    
            r[2]*Cx,
            r[3]*Cx]   

Co=[0.001, 120/30,0,0]                     # [Cxo, Cso, Cgo, Ceo] in cmol/L
tmax=90                                    # Integrate for 20 hours  
tspan=numpy.linspace(0,tmax,200)           # define the timespan of the integration
Ci = odeint(ferm_eq, Co, tspan)             # integration via odeint that calls apon ferm_eq

Cx=Ci[:,0]                                  
Cs=Ci[:,1]
Cg=Ci[:,2]
Ce=Ci[:,3]

plt.figure(2)
plt.plot(tspan, Ce, color='blue', label='e')
plt.plot(tspan, Cg, color='cyan', label='g')
plt.plot(tspan, Cs, color='red',label='s')
plt.plot(tspan, Cx, color='green',label='x')
plt.legend(loc='best')
plt.ylabel('Concentration cmol/L') 
plt.xlabel('time (h)') 
plt.show()

plt.figure(3)
plt.plot(tspan, Ce, color='blue', label='e')
plt.plot(tspan, Cx, color='green',label='x')
plt.legend(loc='best')
plt.ylabel('Concentration cmol/L') 
plt.xlabel('time (h)') 
plt.show()

plt.figure(4)
plt.plot(tspan, Cg, color='cyan', label='g')
plt.legend(loc='best')
plt.ylabel('Concentration cmol/L') 
plt.xlabel('time (h)') 
plt.show()

r=numpy.asarray([response_fun(C) for C in Ci])  #make sure you understand how this 'loop' works

re=r[:,3]
rev=re*Cx
#Let's try the new plotting options
get_ipython().magic('matplotlib notebook')
plt.figure(5)
plt.plot(tspan, re, color='blue', label='re')
plt.legend(loc='best')
plt.ylabel('rate cmol/cmol X /h') 
plt.xlabel('time (h)') 
plt.show()

plt.figure(6)
plt.plot(tspan, rev, color='red', label='re_v')
plt.legend(loc='best')
plt.ylabel('rate cmol/L/h') 
plt.xlabel('time (h)') 
plt.show()

rs=-r[:,1]
Yse=re/rs*23/30; #g/g

plt.figure(7)
plt.plot(tspan, Yse, color='red', label='YSE')
plt.ylabel('yield g/g') 
plt.xlabel('time (h)') 
plt.show()

tmax=40                                    
tspan2=numpy.linspace(0,tmax,200)           
Ci2 = odeint(ferm_eq, Co, tspan2)

r2=response_fun(Ci2[-1,:])     #Last row of concnetration matrix
rev2=r2[3]*Ci2[-1,0]
Yse2=-r2[3]/r2[1]*23/30

[rev2, Yse2]

tdone=numpy.interp(0.0001,Cs[::-1],tspan[::-1]) 
P=max(Ce)/tdone*23
Y=max(Ce)/(120/30)*23/30
[P, Y]



