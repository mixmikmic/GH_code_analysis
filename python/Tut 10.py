import numpy
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')
from scipy.integrate import odeint
alpha, gamma, mu_max, theta_max, PO = 0.1, 2.5, 0.3, 0.1, 1.5
Km_mu, Km_theta = 30/1000/30, 5/1000/30
Mx=12+1.8+0.5*16+0.2*14
V, Cxo, Cso = 200, 0.025/Mx, 150/30

kla=200
Co_s_max=7/1000/32  #mol/l

def response_fun(C):
    
    Cx, Cs = C          
    mu=mu_max*Cs/(Km_mu+Cs)
    theta=theta_max*Cs/(Km_theta+Cs)
    
    ro_mt=kla*Co_s_max
    
    Sbase = numpy.matrix([[-1,1.1,1,0,0],   #Stoichiometric matrix without specification (4x5)
                         [0,0.1,1,0,-1],
                         [0,0.1,2,-2,0],
                         [0,-2.5,2/3,3,0]])
    
    S1 = numpy.vstack([Sbase,[0,1,0,0,0]])  #Stoich matrix with growth spec
      
    C= numpy.matrix([[0,0,0,theta, mu]]).T
    r = numpy.linalg.solve(S1,C)            #rates for regime where no oxygen limitation exist
    
        
    if ro_mt > r[3,0]*Cx:     #no oxygen limitation
        
        return [r[1,0],      #biomass 
                -r[0,0],
                r[4,0],      # CO2 formation
                r[3,0]]      # O2 consumption                                                 
     
    else:
        
        S2 = numpy.vstack([Sbase,[0,0,0,1,0]])          #Stoich matrix with oxygen spec

        C2= numpy.matrix([[0,0,0,theta,ro_mt/Cx]]).T   # Note the ro specification!!
        r2 = numpy.linalg.solve(S2,C2)                 # rates for regime where oxygen limitation exist
        
        
        return [r2[1,0],
                -r2[0,0],
                r2[4,0],    #CO2 formation
                r2[3,0]]    #O2 consumption  

def ferm_eq(C,t):
    Cx, Cs = C[0],C[1]
    r=response_fun([Cx, Cs])     
    
    return [r[0]*Cx,
            r[1]*Cx]    
            

Co=[Cxo, Cso]                     # [Cxo, Cso, Cgo, Ceo] in cmol/L
tmax=120                                   # Integrate for 20 hours  
tspan=numpy.linspace(0,tmax,200)           # define the timespan of the integration
Ci = odeint(ferm_eq, Co, tspan)             # integration via odeint that calls apon ferm_eq

Cx=Ci[:,0]                                  
Cs=Ci[:,1]


plt.figure(2)
plt.plot(tspan, Cx, color='blue', label='Cx')
plt.plot(tspan, Cs, color='red', label='Cs')
plt.legend(loc='best')
plt.ylabel('Concentration cmol/L') 
plt.xlabel('time (h)') 
plt.show()



