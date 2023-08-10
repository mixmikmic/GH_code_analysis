get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *

# costants
#-------------------------------------------------------
n = 200
dt = 0.001
tau = 0.01
epsilon = 1e-60
alpha = 0.1
target = 1-epsilon
itarget = 1.0 - (epsilon/2.0)
#-------------------------------------------------------


# utils
#-------------------------------------------------------
def get_rho_and_eig(X) :
    e = eigvals(X) 
    return  max(abs(e)), e

def get_W_from_V(X) :
    return (tau/dt)*(X - (1-(dt/tau))*eye(n,n)) 

def get_V_from_W(X) :
    return (dt/tau)*X + (1-(dt/tau))*eye(n,n)

def modulate_rotation_contraction(X) :
    W1 = .5*(X-X.T)
    W2 = .5*(X+X.T)

    return alpha*W2 +(1-alpha)*W1

#-------------------------------------------------------
# create initial weights
W = randn(n, n)
W = modulate_rotation_contraction(W)
rho, e = get_rho_and_eig(W) 
3
SSW = W.copy()
# store weights scaled tho rho=1
SW = (W/rho)
print "done"

V = get_V_from_W(W)
rho, e = get_rho_and_eig(V) 
V = (1-epsilon/2.0)*V/rho
W = get_W_from_V(W)



figure("Method1", figsize=(6,2))
#*****************************
m_rho, m_e = get_rho_and_eig(V) 
#*****************************
subplot(121, aspect="equal")
title("eigenvalues of V")
scatter(real(m_e), imag(m_e))
xlim([-m_rho, m_rho])
ylim([-m_rho, m_rho])
#*****************************

#*****************************
rho, e = get_rho_and_eig(W) 
#*****************************
subplot(122, aspect="equal")
title("eigenvalues of W")
scatter(real(e), imag(e))
xlim([-rho, rho])
ylim([-rho, rho])
#*****************************

print """
ESP: analytical method (the wrong one)"

W
--------------------------------------
rho := {:16.12f}  
min := {:8.4f}  max := {:8.4f}   

V
--------------------------------------
rho := {:16.12f}  
min := {:8.4f}  max := {:8.4f}   
""".format(rho, W.min(), W.max(), 
           m_rho, V.min(), V.max())

from scipy.optimize import bisect

def dist(rho_estimate) :
    V = get_V_from_W(rho_estimate*SW)
    effective_rho_estimate,_ = get_rho_and_eig(V)
    return itarget - effective_rho_estimate

import time
start = time.clock()

# ---------------------------------------------------------------
rho_estimate = bisect(lambda x: dist(x), 
                      1-epsilon, tau/dt, 
                      xtol = epsilon/2.)
# ---------------------------------------------------------------

print
print
print "computed in {}s".format(time.clock() - start)
print 
print

W = SW*rho_estimate
V = get_V_from_W(W)

figure("Method2")
#*****************************
m_rho, m_e = get_rho_and_eig(V) 
#*****************************
subplot(121, aspect="equal")
title("eigenvalues of V")
scatter(real(m_e), imag(m_e))
xlim([-m_rho, m_rho])
ylim([-m_rho, m_rho])
#*****************************

#*****************************
rho, e = get_rho_and_eig(W) 
#*****************************
subplot(122, aspect="equal")
title("eigenvalues of W")
scatter(real(e), imag(e))
xlim([-rho, rho])
ylim([-rho, rho])
#*****************************

print """
ESP: iterative method
W
--------------------------------------
rho := {:16.12f}  
min := {:8.4f}  max := {:8.4f}   

V 
--------------------------------------
rho := {:16.12f}  
min := {:8.4f}  max := {:8.4f}   
""".format(rho, W.min(), W.max(), 
           m_rho, V.min(), V.max())

from IPython.display import Image
Image('pics/intersection.png', width=500)

target = 1 - epsilon/2.

W = SW.copy()

#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
start = time.clock()

# --------------------------------------------------

rho, e = get_rho_and_eig(W) 
x = e.real
y = e.imag
h = dt/tau
a = x**2*h**2 + y**2*h**2
b = 2*x*h - 2*x*h**2
c = 1 + h**2 - 2*h - target**2

# just get the positive solutions
sol2 = (-b + sqrt(b**2 - 4*a*c))/(2*a)
# and take the minor amongst them
correctRhoMultiplier = min(sol2)
W = correctRhoMultiplier*W
# --------------------------------------------------
print
print
print "computed in {}s".format(time.clock() - start)
print 
print

#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

V = get_V_from_W(W)

figure("Method3")
#*****************************
m_rho, m_e = get_rho_and_eig(V)
#*****************************
subplot(121, aspect="equal")
title("eigenvalues of M_tilde")
scatter(real(m_e), imag(m_e))
xlim([-m_rho, m_rho])
ylim([-m_rho, m_rho])
#*****************************

#*****************************
rho, e = get_rho_and_eig(W)
#*****************************
subplot(122, aspect="equal")
title("eigenvalues of W")
scatter(real(e), imag(e))
xlim([-rho, rho])
ylim([-rho, rho])
#*****************************
print """
ESP: analytical method (the RIGHT one)
W
--------------------------------------
rho := {:16.12f}  
min := {:8.4f}  max := {:8.4f}   

V 
--------------------------------
rho := {:16.12f}  
min := {:8.4f}  max := {:8.4f}   
""".format(rho, W.min(), W.max(), 
           m_rho, V.min(), V.max())

T = 100
n_test = 20
u = zeros([n,T])
u[:,:10] = np.random.rand(n,10)

x_f = np.zeros((n,T))
x_j = np.zeros((n,T))
x_f[:,0] = 0
x_j[:,0] = 0
for t in range(1,T):
    x_f[:,t] = x_f[:,t-1]-h*x_f[:,t-1]+h*(u[:,t]+W.dot(tanh(x_f[:,t-1])))
    x_j[:,t] = x_j[:,t-1]-h*x_j[:,t-1]+tanh(u[:,t]+h*W.dot(x_j[:,t-1]))

figure("ESN Comparison: same dynamics but 1/h times wider h = %f" % h, figsize=(15,5))
subplot(121)#, aspect="equal")
title("Evolution of our ESN (first %i neurons)" % n_test)
plot(x_f[0:n_test,:].T)

subplot(122)#, aspect="equal")
title("Evolution of Jaeger's ESN (first %i neurons)" % n_test)
plot(x_j[0:n_test,:].T)

tight_layout()

