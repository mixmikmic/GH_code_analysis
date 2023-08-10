from __future__ import division, print_function
import math
import numpy as np
def KM_P_value(n, o1, o2, u1, u2, gamma, mu):
    return((1 - mu/(2*gamma))**n *           (1 - 1/(2*gamma))**(-o1) *           (1 - 1/gamma)**(-o2) *           (1 + 1/(2*gamma))**(-u1) *           (1 + 1/gamma)**(-u2))

def KM_Expected_sample_size(or1, or2, ur1, ur2, mu, gamma, alpha):
    n = np.nan
    denom = np.log( 1 - mu/(2*gamma) ) -            or1*np.log(1 - 1/(2*gamma)) -            or2*np.log(1 - 1/gamma) -            ur1*np.log(1 + 1/(2*gamma)) -            ur2*np.log(1 + 1/gamma)
    if (denom < 0):
        n = np.ceil(np.log(alpha)/denom)
    return(n)

alpha = 0.05
gamma = 1.03905  

mu = (354040 - 337589)/(354040+337589+33234) # New Hampshire 2016
KM_P_value(200, 1, 0, 0, 0, gamma, mu)

KM_Expected_sample_size(.001,0,.001,0,0.05, gamma, alpha)

KM_Expected_sample_size(.05,0,0,0,0.05, gamma, alpha)



