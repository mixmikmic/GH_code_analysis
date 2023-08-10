from math import pi
import numpy as np
from math import sin, cos, acos, exp, sqrt


def inductionFactors(r, chord, Rhub, Rtip, phi, cl, cd, B,
        Vx, Vy, useCd, hubLoss, tipLoss, wakerotation):
    """Computes induction factors and residual error at a given location
    on the blade.  Full details on inputs/outputs ommitted here."""

    sigma_p = B/2.0/pi*chord/r
    sphi = sin(phi)
    cphi = cos(phi)

    # resolve into normal and tangential forces
    if not useCd:
        cn = cl*cphi
        ct = cl*sphi
    else:
        cn = cl*cphi + cd*sphi
        ct = cl*sphi - cd*cphi

    # Prandtl's tip and hub loss factor
    Ftip = 1.0
    if tipLoss:
        factortip = B/2.0*(Rtip - r)/(r*abs(sphi))
        Ftip = 2.0/pi*acos(exp(-factortip))

    Fhub = 1.0
    if hubLoss:
        factorhub = B/2.0*(r - Rhub)/(Rhub*abs(sphi))
        Fhub = 2.0/pi*acos(exp(-factorhub))

    F = Ftip * Fhub

    # bem parameters
    k = sigma_p*cn/4.0/F/sphi/sphi
    kp = sigma_p*ct/4.0/F/sphi/cphi

    # compute axial induction factor
    if phi > 0.0:  # momentum/empirical

        # update axial induction factor
        if k <= 2.0/3.0:  # momentum state
            a = k/(1+k)

        else:  # Glauert(Buhl) correction

            g1 = 2.0*F*k - (10.0/9-F)
            g2 = 2.0*F*k - (4.0/3-F)*F
            g3 = 2.0*F*k - (25.0/9-2*F)

            if abs(g3) < 1e-6:  # avoid singularity
                a = 1.0 - 1.0/2.0/sqrt(g2)
            else:
                a = (g1 - sqrt(g2)) / g3


    else:  # propeller brake region (a and ap not directly used but update anyway)

        if k > 1.0:
            a = k/(k-1.0)
        else:
            a = 0.0  # dummy value

    # compute tangential induction factor
    ap = kp/(1.0-kp)

    if not wakerotation:
        ap = 0.0
        kp = 0.0

    # error function
    lambda_r = Vy/Vx
    if phi > 0:  # momentum/empirical
        fzero = sphi/(1.0-a) - cphi/lambda_r*(1.0-kp)
    else:  # propeller brake region
        fzero = sphi*(1.0-k) - cphi/lambda_r*(1.0-kp)

    return fzero, a, ap

# wrap function
def function(x, params):
    
    # unpack variables
    r, chord, Rhub, Rtip, phi, cl, cd, Vx, Vy = x
    B, useCd, hubLoss, tipLoss, wakerotation = params

    # call the original function
    return inductionFactors(r, chord, Rhub, Rtip, phi, cl, cd, B,
        Vx, Vy, useCd, hubLoss, tipLoss, wakerotation)

# setup inputs
r = 0.5
chord = 0.1
Rhub = 0.1
Rtip = 1.0
phi = 0.2
cl = 0.3
cd = 0.002
B = 3
Vx = 1.0
Vy = 5.0
useCd = True
hubLoss = True
tipLoss = True
wakerotation = True

x = np.array([r, chord, Rhub, Rtip, phi, cl, cd, Vx, Vy])
params = np.array([B, useCd, hubLoss, tipLoss, wakerotation])

n = len(x)

# ------ finite difference --------
output, a, ap = function(x, params)  # we are ignoring the other outputs although we could easily get there derivatives as well

g_fd = np.zeros(n)  # initialize gradient vector for finite difference

for i in range(n):  # iterate across all vars

    # step size
    step = 1e-6*x[i]  

    # take a step
    xplus = np.copy(x)  
    xplus[i] += step

    output_plus, a, ap = function(xplus, params)

    g_fd[i] = (output_plus - output) / step

# You can ignore this.  I'm just ignoring a printed FutureWarning about a change affecting something internal to algopy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

from algopy import UTPM  # just the name of the algorithm (stands for univariate Taylor propagation of matrices)
from algopy import sin, cos, exp, sqrt  # overloaded versions of functions we use
from algopy import arccos as acos  # need to rename b.c. using the math version (acos) whereas numpy uses arccos

# create an algopy version of x
x_algopy = UTPM.init_jacobian(x)

# create an algopy version of outputs
output, a, ap = function(x_algopy, params)

# extract the gradients
g_ad_oo = UTPM.extract_jacobian(output)  # could call again for the other outputs

from _bem import inductionfactors_dv

# get derivative of each input
I = np.eye(n)
dr = I[0, :]
dchord = I[1, :]
dRhub = I[2, :]
dRtip = I[3, :]
dphi = I[4, :]
dcl = I[5, :]
dcd = I[6, :]
dVx = I[7, :]
dVy = I[8, :]


fzero, a, ap, doutput_dx, da_dx, dap_dx = inductionfactors_dv(r, chord, Rhub, Rtip,
    phi, cl, cd, B, Vx, Vy, dr, dchord, dRhub, dRtip, dphi, dcl, dcd, dVx, dVy)

# rename the gradient
g_ad_sc = doutput_dx

# import complex versions
from cmath import sin, cos, acos, exp, sqrt

# redine absolute value
def c_abs(x):
    if x.real < 0:
        return -x
    else:
        return x

abs = c_abs

# initialize
g_cs = np.zeros(n)

# iterate across entires in x
for i in range(n):
    step_complex = 1e-30  # take a really small step

    # new xvalue: x + ih
    xcomplex = np.copy(x).astype(complex)
    xcomplex[i] += complex(0.0, step_complex)

    # call function
    output_complex, a_complex, ap_complex = function(xcomplex, params)

    # compute gradient
    g_cs[i] = output_complex.imag / step_complex

from __future__ import print_function

print('error_fd =', (g_fd - g_cs)/g_cs)

print('error_ad_oo =', (g_ad_oo - g_cs)/g_cs)
print('error_ad_sc =', (g_ad_sc - g_cs)/g_cs)



