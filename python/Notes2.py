# Scipy stuff
from sympy import Symbol, symbols, init_printing, oo
from sympy import simplify
from sympy import exp, gamma, log, LambertW
from sympy.integrals import integrate
from sympy import *
from scipy.optimize import fsolve

# From future
from __future__ import division

# Setting things up
init_printing(use_unicode = False, wrap_line = False, no_global = True)
#variables
k, wp, ws, lambda_int, beta, alpha, m, kappa,ep = symbols('k wp m lambda_\mathrm{int} beta alpha m kappa ep', real = True, positive = True)





#since 1-ps=ep

#T = (log(1+ beta,2)/mbar)*(ep**(1+m))
#T = (log(1+ beta,2))*(1-(ep**(1+m)))
#T=(1-(ep**(1+m)))*log(1+beta)
#be=((1-exp(-k*beta**(2/alpha)))**(1+m))


   
beta=((-ln(1-ep**(1/(1+m))))/k)**(alpha/2)
T=(1-ep**(1/(1+m)))*log(1+beta)
T

h=diff(T,m)
h

#simplifying the derivative 

z1=alpha*((-1/k)**(alpha/2))*(((log(1-ep**(1/(1+m)))))**((alpha/2)-1))
z2=2*((-1/k)*log(1-ep**(1/(1+m)))**(alpha/2)+1)
z3=log((-1/k)*((log(1-ep**(1/(1+m))))))
sim=(z1/z2)+z3
sim

alpha=4
d=1
lambda_int=0.05
kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
k=kappa*lambda_int*3.14*d**2
ep=0.02
#m=5

z1=alpha*((-1/k)**(alpha/2))*(((log(1-ep**(1/(1+m)))))**((alpha/2)-1))
z2=2*((-1/k)*log(1-ep**(1/(1+m)))**(alpha/2)+1)
z3=log((-1/k)*((log(1-ep**(1/(1+m))))))
sim=(z1/z2)+z3
sim

#plotting
get_ipython().magic('matplotlib notebook')
from sympy import Symbol, symbols, init_printing, oo
from sympy import simplify
from sympy import exp, gamma, log, LambertW
from sympy.integrals import integrate
from sympy import *
from scipy.optimize import fsolve

# From future

from matplotlib import rc, rcParams
from scipy.optimize import fsolve
from __future__ import division
import numpy as np
import math

import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

# Setting things up
init_printing(use_unicode = False, wrap_line = False, no_global = True)
#variables
k, wp, ws, lambda_int, beta, alpha, m, kappa,ep = symbols('k wp m lambda_\mathrm{int} beta alpha m kappa ep', real = True, positive = True)

T=[]


for m in range(0,25):
    alpha=4
    d=1
    lambda_int=0.05
    kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
    k=kappa*lambda_int*np.pi*d**2
    ep=0.02
    #alpha=3
    #kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
  

    beta=((-ln(1-ep**(1/(1+m))))/k)**(alpha/2)
   

     
    T.append((1-ep**(1/(1+m)))*log(1+beta))
    #T.append((((1-ep**(1/(1+m)))*log(1+beta))*ep)/(1-ep))
    #print(T)
   
fig = plt.figure()
fig.suptitle('alpha=4,epsilon=0.02', fontsize=10)    
plt.xlabel('m')
plt.ylabel('T')
plt.plot(range(0,25),T, marker = 'o', linestyle = 'solid', label='Throughput')

plt.legend(loc = 'best')
plt.show()

T

#plotting for different lamdas
from IPython.display import Latex
from sympy import Symbol, symbols, init_printing, oo
from sympy import simplify
from sympy import exp, gamma, log, LambertW
from sympy.integrals import integrate
from sympy import *
from scipy.optimize import fsolve

# From future

from matplotlib import rc, rcParams
from scipy.optimize import fsolve
from __future__ import division
import numpy as np
import math

import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

# Setting things up
init_printing(use_unicode = False, wrap_line = False, no_global = True)
#variables
k, wp, ws, lambda_int, beta, alpha, m, kappa,ep = symbols('k wp m lambda_\mathrm{int} beta alpha m kappa ep', real = True, positive = True)

T=[]


for m in range(0,25):
    alpha=4
    d=1
    lambda_int=0.05
    kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
    k=kappa*lambda_int*np.pi*d**2
    ep=0.02
    #alpha=3
    #kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
  

    beta=((-ln(1-ep**(1/(1+m))))/k)**(alpha/2)
   

     
    T.append((1-ep**(1/(1+m)))*log(1+beta))
#############################################################
    
T1=[]
for m in range(0,25):
    alpha=4
    d=1
    lambda_int=0.10
    kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
    k=kappa*lambda_int*pi*d**2
    ep=0.02
    #alpha=3
    #kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
  

    beta=((-ln(1-ep**(1/(1+m))))/k)**(alpha/2)
   

     
    T1.append((1-ep**(1/(1+m)))*log(1+beta))
###################################################################

T2=[]
for m in range(0,25):
    alpha=4
    d=1
    lambda_int=0.15
    kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
    k=kappa*lambda_int*pi*d**2
    ep=0.02
    #alpha=3
    #kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
  

    beta=((-ln(1-ep**(1/(1+m))))/k)**(alpha/2)
   

     
    T2.append((1-ep**(1/(1+m)))*log(1+beta))
##################################################################
T3=[]
for m in range(0,25):
    alpha=4
    d=1
    lambda_int=0.20
    kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
    k=kappa*lambda_int*pi*d**2
    ep=0.02
    #alpha=3
    #kappa = gamma(1 + 2/alpha)* gamma(1- 2/alpha)
  

    beta=((-ln(1-ep**(1/(1+m))))/k)**(alpha/2)
   

     
    T3.append((1-ep**(1/(1+m)))*log(1+beta))
##################################################################
get_ipython().magic('matplotlib notebook')
fig = plt.figure()
fig.suptitle('alpha=4,epsilon=0.02', fontsize=10)    
plt.xlabel('m')
plt.ylabel('T')
plt.plot(range(0,25),T, marker = 'o', linestyle = 'solid', label='Throughput,$\lambda$=0.05')
plt.plot(range(0,25),T1, marker = 'o', linestyle = 'solid', label='Throughput,$\lambda$=0.10')
plt.plot(range(0,25),T2, marker = 'o', linestyle = 'solid', label='Throughput,$\lambda$=0.15')
plt.plot(range(0,25),T3, marker = 'o', linestyle = 'solid', label='Throughput,$\lambda$=0.20')

plt.legend(loc = 'best')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

points = np.array([(5, 1.09), (6, .6),(9, .38), (11, .27)])
# get x and y vectors
x = points[:,0]
y = points[:,1]

fig = plt.figure()
axes = plt.gca()
axes.set_xlim([4,12])
axes.set_ylim([0,1.2])

get_ipython().magic('matplotlib notebook')
plt.plot(x, y)
plt.yscale('linear')
plt.grid(True)

plt.title('Effect of increasing the packet density on the throughput')
fig.suptitle('alpha=4,epsilon=0.02', fontsize=10)    
plt.xlabel('m')
plt.ylabel('T')
plt.plot(x, y, '-gD')

plt.annotate('$\lambda=0.05$',
            xy=(5, 1.09),  # theta, radius
            xytext=(.4, .8),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='bottom',
            )

plt.annotate('$\lambda=0.1$',
            xy=(6, .6),  # theta, radius
            xytext=(.5, .5),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='bottom',
            )

plt.annotate('$\lambda=0.15$',
            xy=(9, .38),  # theta, radius
            xytext=(.7, .4),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='bottom',
            )

plt.annotate('$\lambda=0.20$',
            xy=(11, .27),  # theta, radius
            xytext=(.54, .15),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='bottom',
            )
plt.show()
    
   



