from __future__ import division
import numpy as np
from math import pi

#variable delaration
mu_0 = 4*pi*10**(-7)  #  permeability in free space
mu_r1 = 3  #  region 1 relative permeability
mu_r2 = 5  #  region 2 relative permeability
mu_1 = mu_r1*mu_0  #  region 1 permeability
mu_2 = mu_r2*mu_0  #  region 2 permeability

#calculations
H1  = np.array([4,1.5,-3])  #  magnetic field in region 1 in A/m
Ht1 = np.array([0,1.5,-3])  #  tangential component of magnetic field H1
Hn1 = np.array([4,0,0])     #  normal component of magnetic field H1
Ht2 = np.array([0,1.5,-3])  #  as tangential componenet of magnetic field H2 = tangential component of magnetic field H1
Hn2 = (mu_1/mu_2)*Hn1       #  normal component of magnetic field H2
H2  = Ht2+Hn2               #  magnetic field in region 2 in A/m
h2  = np.linalg.norm(H2)    #  magnitude of the magnetic field H2 in A/m

#results
print "magnetic field in region 2 in A/m:",np.around(H2,2)
print "magnitude of magnetic field in region 2 in A/m:",round(h2,3) 

from __future__ import division
import numpy as np

#variable Declaration
epsilon_0 = 8.854*10**(-12) # permittivity in free space
sigma_1 = 0 #conductivity of medium 1
sigma_2 = 0 #conductivity of medium 2
epsilon_r1 = 1 # region 1 relative permittivity
epsilon_r2 = 2 # region 2 relative permittivity

#calculations
epsilon_1 = epsilon_r1*epsilon_0 # region 1 permittivity
epsilon_2 = epsilon_r2*epsilon_0 # region 2 permittivity
E1  = np.array([1,2,3])  # Electric field in region 1 in V/m
Et1 = np.array([0,2,3])  # tangential component of electric field E1
En1 = np.array([1,0,0])  # normal component of electric field E1
Et2 = np.array([0,2,3])  # as tangential componenet of electric field E2 = tangential component of electric field E1
En2 = (epsilon_1/epsilon_2)*En1 # normal component of electric field E2
E2  = Et2+En2 # electric field in region 2 in V/m
Dt1 = epsilon_0*Et1  # tangential component of electric flux density D1
D2  = epsilon_2*E2   # electric flux density in region 2 in C/m**2


#Results
print "electric field in region 2 in V/m:",np.around(E2,2)
print "electric flux density in region 2 in C/m**2:",D2 

from __future__ import division
import numpy as np
from math import pi

#variable Declaration

#  H = cos(10**8*t-Beta*z)ay       #  magnetic field in A/m
#  E = 377*cos(10**8*t-Beta*z)ax   #  electric field in V/m
omega = 10**8    #  angular frequency in Hz
v_0 = 3*10**8    #  speed of light in m/s


#calculations
f = omega/(2*pi) #  frequency in Hz
lamda = v_0/f    #  wavelength in m
Beta = (2*pi)/lamda #  phase constant in rad/m
print "eta_0 =  E/H  =  377*cos(10**8*t-Beta*z)/cos(10**8*t-Beta*z)  = > E/H = 377"
eta_0 = abs(377) #  intrinsic impedence in ohm



#Results
print "intrinsic impedence in ohm:",eta_0
print "frequency in MHz:",round(f/(10**6),3)
print "phase constant in rad/m:",round(Beta,3)
print "wavelength in m:",round(lamda,3)

from __future__ import division
from math import pi
import numpy as np

#Variable Declaration
f = 100        #  frequency in MHz
f = 100*10**6  #  frequency in Hz
v_0=3*10**8    #  speed of light in m/s


#  formula : Gamma = omega(j)*sqrt(mu_0*epsilon_0)=omega(j)/v_0 =(2j*pi*f)/v_0
Gamma =(2j*pi*f)/(v_0)  # propagation constant


#result
print "propagation constant in m**-1:",np.around(Gamma,3)

from __future__ import division
from math import pi

#Variable Declaration

# H(z,t) =  48*cos(10**8*t+40*z)ay  #  equation of magnetic field 
A = 48  #  amplitude of the magnetic field in A/m
omega = 10**8  #  angular frequency in radians/sec
Beta = 40  #  phase constant in rad/m

#Calculations
f = omega/(2*pi)  #  frequency in Hz
lamda = (2*pi)/Beta  #  wavelength in m


#results
print "amplitude of the magnetic field in A/m:",A
print "frequency in MHz:",round(f/10**6,3)
print "phase constant in rad/m:",round(Beta,3)
print "wavelength in m:",round(lamda,3)

from __future__ import division
from math import pi,sqrt

#Variable Declaration

H = 2  #  ampliutude of magnetic field in A/m
sigma = 0  #  conductivity
mu_0 = 4*pi*10**-7  #  permeability in free space in H/m
epsilon_0 = 8.854*10**-12  #  permittivity in free space in F/m

#calculations
mu = mu_0  #  permeability in F/m
epsilon = 4*epsilon_0  #  permittivity in F/m
Eta_0 = 120*pi  #  intrinsic impedence in free space in ohm
E_free = Eta_0*H  #  electric field in V/m


#results
print "magnitude of electric field in V/m in free space:",round(E_free,3)
Eta = sqrt(mu/epsilon)  #  intrinsic impedence in ohm
E = Eta*H  #  magnitude of electric field
print "magnitude of electric field in V/m:",round(E,3)

from __future__ import division
from math import pi,sqrt
import numpy as np


#variable Declaration

sigma = 0  #  conductivity in mho/m
f = 0.3  #  frequency in GHz
f = 0.3*10**9  #  frequency in Hz
omega = 2*pi*f  #  angular frequency in rad/sec
 #  formula : Gamma = sqrt(1j*omega*mu*(sigma+1j*omega*epsilon)) = 1j*omega*sqrt(mu*epsilon)
epsilon_0 = 8.854*10**-12  #  permittivity in free space in F/m
epsilon = 9*epsilon_0  #  permittivity in F/m
mu_0 = 4*pi*10**-7  #   permeability in free space in H/m
mu = mu_0  #  permeability in H/m
Gamma = 1j*omega*sqrt(mu*epsilon)  #  propagation constant im m**-1


#results

print "propagation constant im m**-1:",np.around(Gamma,3)
 #  formula : eta = sqrt((1j*omega*mu)/(sigma+omega*epsilon)) = sqrt(mu/epsilon)
eta = sqrt(mu_0/(9*epsilon_0))  #  intrinsic impedence in ohm
print "intrinsic impedence in ohm:",round(eta,3)



# note : answer in the book is wrong.



from __future__ import division
from math import sqrt,pi

#variable declaration

lamda = 0.25    #  wavelength in m
v = 1.5*10**10  #  velocity of propagation of wave in cm/sec
v = 1.5*10**8   #  velocity of propagation of wave in m/sec
epsilon_0 = 8.854*10**-12  #  permittivity in free space in F/m
mu_0 = 4*pi*10**-7         #   permeability in free space in H/m
mu = mu_0      #  permeability in H/m
v_0 = 3*10**8  #  speed of light in m/s
f = v/lamda     #  frequency in Hz
 #  formula : v = 1/(mu*epsilon) = 1/(mu_0*epsilon_0*epsilon_r) = v_0/sqrt(epsilon_r)
epsilon_r = (v_0/v)**2  #  relative permittivity


#results
print "frequecy in MHz:",round(f/10**6,3)
print "relative permittivity:",epsilon_r


 # note : answer in the book is wrong.

from __future__ import division
from math import sqrt,pi

#variable declaration and calculations

#E = 5*sin(10**8*t+4*x)az  #  equation of electric field 

A = 5  #  amplitude of the electric field
omega = 10**8  #  angular frequency in radians/sec
f = omega/(2*pi)  #  frequency in Hz
Beta = 4  #  phase constant in rad/m
v_0 = 3*10**8  #  speed of light in m/s
lamda = v_0/f  #  wavelength in m


#results
print "frequency in MHz:",round(f/10**6,3)
print "phase constant in rad/m:",round(Beta,3)
print "wavelength in m:",round(lamda,3)

from __future__ import division
from math import pi

sigma = 10**-2  #  conductivity of earth in mho/m
epsilon_r = 10  #  relative permittivity
mu_r = 2  #  relative permeability
epsilon_0 = (1/(36*pi))*10**-9  #  permittivity in free space
epsilon = epsilon_r*epsilon_0  #  permittivity
f1 = 50  #  frequency in Hz
omega1 = 2*pi*f1  #  angular frequency in rad/sec
print "When frequency = 50Hz:"
k1 = sigma/(omega1*epsilon)
print "K1 is equal to",k1
print "since k1>>1 hence it behaves like a good conductor:"
f2 = 1  #  frequency in kHz
f2 = 1*10**3  #  frequency in Hz
omega2 = 2*pi*f2  #  angular frequency in rad/sec
print "When frequency = 1kHz:"
k2 = sigma/(omega2*epsilon)
print "K2 is equal to",k2
print "since k2>>1 hence it behaves like a good conductor:"
f3 = 1  #  frequency in MHz
f3 = 1*10**6  #  frequency in Hz
omega3 = 2*pi*f3  #  angular frequency in rad/sec
print "When frequency = 1MHz:"
k3 = sigma/(omega3*epsilon)
print "K3 is equal to",k3
print "since k3 = 18 hence it behaves like a moderate conductor:"
f4 = 100  #  frequency in MHz
f4 = 100*10**6  #  frequency in Hz
omega4 = 2*pi*f4  #  angular frequency in rad/sec
print "When frequency = 100MHz:"
k4 = sigma/(omega4*epsilon)
print "K4 is equal to",k4
print "since k4 = 0.18 hence it behaves like a quasi-dielectric:"
f5 = 10  #  frequency in GHz
f5 = 10*10**9  #  frequency in Hz
omega5 = 2*pi*f5  #  angular frequency in rad/sec
print "When frequency = 10GHz:"
k5 = sigma/(omega5*epsilon)
print "K5 is equal to",k5
print "since k5<<1 hence it behaves like a good dielectric:"

from __future__ import division
from math import sqrt,pi
import cmath
import numpy as np

#variable declaration
f = 60  # frequency in Hz
omega = 2*pi*f  # angular frequency in rad/sec
sigma = 5.8*10**7  # conductivity in mho/m
epsilon_0 = 8.854*10**-12  # permittivity in free space in F/m
mu_0 = 4*pi*10**-7  #  permeability in free space in H/m
epsilon_r = 1  # relative permittivity
mu_r = 1  # relative permeability


#calculations
epsilon = epsilon_r*epsilon_0  # permittivity
mu = mu_0*mu_r  # permeability
k = sigma/(omega*epsilon)  # ratio
print "ratio k is equal to",k
print "since k>>1 therefore it is very good conductor:"
alpha = sqrt(omega*mu*sigma/2)  # attenuation constant in m**-1
Beta = sqrt(omega*mu*sigma/2)  # phase constant in m**-1
Gamma = alpha+(1j*Beta)  # propagation constant in m**-1
lamda = (2*pi)/Beta  # wavelength
eta = cmath.sqrt(((1j*omega*mu)/sigma))  # intrinsic impedence in ohm
v = lamda*f  # phase velocity of wave in m/s


#result
print "attenuation constant in m**-1:",round(alpha,2)
print "phase constant in m**-1:",round(Beta,2)
print "propagation constant in m**-1:",np.around(Gamma,2)
print "intrinsic impedence in ohm:",np.around(eta,10)
print "wavelength in cm:",round(lamda*100,2)
print "phase velocity of wave in m/s:",round(v,3)

from __future__ import division
from math import sqrt,pi


#variable Declaration

f1 = 60  #  frequency in Hz
omega1 = 2*pi*f1  #  angular frequency in Hz
f2 = 100          #  frequency in MHz
f2 = 100*10**6    #  frequency in Hz
omega2 = 2*pi*f2  #  angular frequency in Hz
sigma = 5.8*10**7 #  conductivity in mho/m
epsilon_0 = 8.854*10**-12  #  permittivity in free space in F/m
mu_0 = 4*pi*10**-7         #  permeability in free space in H/m
epsilon_r = 1    #  relative permittivity
mu_r = 1         #  relative permeability
epsilon = epsilon_r*epsilon_0  #  permittivity
mu = mu_0*mu_r   #  permeability

print "At f = 60Hz"
k1 = (sigma)/(omega1*epsilon)  #  ratio
print "ratio k is equal to",k1
print "since k>>1 therefore it is very good conductor at f = 60Hz:"
delta1 = (sqrt(2/(omega1*mu*sigma)))  #  depth of penetration in m
print "depth of penetration delta1 in m:",delta1

print "At f = 100Hz"
k2 = sigma/(omega2*epsilon)  #  ratio
print "ratio k is equal to",k2
print "since k2>>1 therefore it is very good conductor at f = 100Hz:"
delta2 = (sqrt(2/(omega2*mu*sigma)))  #  depth of penetration in m
print "depth of penetration delta2 in m:",delta2

from __future__ import division
from math import sqrt,pi


#variable Declaration

Ic = 10  #  conduction current in ampere
epsilon_r = 1  #  relative permittivity
epsilon_0 = 8.854*10**-12  #  permittivity in free space
epsilon = epsilon_r*epsilon_0  #  permittivity
sigma = 5.8*10**7  #  conductivity in mho/m

print "when f = 1MHz"
f = 1  #  frequency in MHz
f = 1*10**6  #  frequency in Hz
Id = (2*pi*f*epsilon*Ic)/sigma  #  printlacement current
print "displacement current when f = 1MHz in A:",Id
print "when f = 100MHz"
f = 100  #  frequency in MHz
f = 100*10**6  #  frequency in Hz
Id = (2*pi*f*epsilon*Ic)/sigma  #  printlacement current
print "displacement current when f = 100MHz in A:",Id

from __future__ import division
from math import sqrt,pi,sin,cos,radians,log

#variable declaration
Em = 20          # minimum signal level required for vessel under sea water in microV/m
Em = 20*10**-6   # minimum signal level required for vessel under sea water in V/m
E = 100          # electric intensity of wave in V/m
v = 3*10**8      # speed of light in m/s
f = 4            # frequency in MHz
f = 4*10**6      # frequency in Hz
omega = 2*pi*f   # angular frequency in Hz
sigma = 4        # conductivity of sea water in mho/m
epsilon_r = 81   # relative permittivity
epsilon_0 = 8.854*10**-12     # permittivity in free space
epsilon = epsilon_r*epsilon_0 # permittivity
mu_r = 1             # relative permeability
mu_0 = 4*pi*10**(-7) # permeability in free space
mu = mu_r*mu_0    # permeability
k = (sigma)/(omega*epsilon)  #ratio
print "ratio k is equal to:"
print "ratio:",round(k,3)
print "K is >>1 so sea water is a good conductor"
eta_1 = 377   # intrinsic impedance in free space in ohm
alpha_1 = 0   # attenuation constant in free space in m**-1


#calculations
beta_1 = omega/v  # phase constant in m**-1
mageta_2 = sqrt((omega*mu)/sigma)   # magnitude of eta_2(intrinsic impedance of sea water in ohm) 
argeta_2 = 45                     # argument of eta_2 in degrees
eta_2 = mageta_2*cos(radians(argeta_2))+(1j*mageta_2*sin(radians(argeta_2)))    #intrinsic impedance in complex form (r*cos(theta)+1j*r*sin(theta))
TC = 2*eta_2/(eta_1+eta_2)          # transmission cofficient
Et = abs(TC)*E                      # transmitted electric field in V/m
alpha_2 = sqrt((omega*mu*sigma)/2)  # attenuation constant for sea water in m**-1
# formula: Et*exp(-alpha_2*d) = Em
d = -(1/alpha_2)*(log(Em/Et))   # depth in the sea that can be reached by the aeroplane in m


#result
print "depth in the sea that can be reached by the aeroplane in m:",round(d,5)


# note 1: the value of alpha_2 in book is 7.905 but it is "7.94" exactly calculated by python.
#note 2 : The correct answer of the Depth(d) is "1.41095" the answer in the book is wrong.

from __future__ import division
from math import sqrt

#variable declaration

eta_0=377  #  intrinsic impedance in free space in ohm
print "E=sin(omega*t-beta*z)ax+2*sin(omega*t-beta*z+75)ay  #  electric field in V/m"
Ex=1  #  magnitude of Ex
Ey=2  #  magnitude of Ey

#calculations
E=sqrt(Ex**2+Ey**2)       #  resultant magnitude
Pav=((1/2)*E**2)/(eta_0)  #  power per unit area conveyed by the wave in free space

#results
print "power per unit area conveyed by the wave in free space in mW/m**2:",round(Pav*1000,3)

from __future__ import division
from math import sqrt,pi

#variable declaration

epsilon_0 = 8.854*10**-12  # permittivity in free space in F/m
mu_0 = 4*pi*10**-7  #  permeability in free space in H/m
epsilon_r = 4  # relative permittivity
mu_r = 1  # relative permeability
epsilon = epsilon_r*epsilon_0  # permittivity
mu = mu_0*mu_r  # permeability
H = 5  # magnitude of magnetic field in mA/m
H = 5*10**-3  # magnitude of magnetic field in A/m

#calculations
eta = sqrt(mu/epsilon) # intrinsic impedence in ohm
E = H*sqrt(mu/epsilon)  # magnitude of electric field
P_av = E**2/(2*eta)  # average power
W_E = epsilon*E**2  # maximum energy density of the wave


#results
print "Average power in micro*w/m**2:",round(P_av*10**6,2)
print "maximum energy density of the wave in PJ/m*3:",round(W_E*10**12,3)


#note: P_av is =  2353.75 in book but it is 2354.58 correctly calculated by python.

from __future__ import division
from math import sqrt,pi

#variable declaration

epsilon_0 = 8.854*10**-12 #  permittivity in free space in F/m
mu_0 = 4*pi*10**-7 #   permeability in free space in H/m
epsilon_r = 1 #  relative permittivity
mu_r = 1 #  relative permeability
epsilon = epsilon_r*epsilon_0 #  permittivity
mu = mu_0*mu_r #  permeability
E = 100*sqrt(pi) #  magnitude of electric field in V/m


#calculations
W_E = (1/2)*epsilon*E**2 #  electric energy density of the wave
W_H = W_E #  as the energy density is equal to that of magnetic field for a pla`ne travelling wave
W_T = W_E+W_H #  total energy density

#results
print "electric energy density of the wave in nJ/m**3:",round(W_E*10**9,3)
print "magnetic energy density of wave in nJ/m**3:",round(W_H*10**9,3)
print "Total energy density in nJ/m**3:",round(W_T*10**9,3)

from __future__ import division
from math import sqrt,pi

#variable Declaration

sigma = 5 #  conductivity of sea water in mho/m
f1 = 25 #  frequency in kHz
f1 = 25*10**3 #  frequency in Hz
omega1 = 2*pi*f1 #  angular frequency in Hz
f2 = 25 #  frequency in MHz
f2 = 25*10**6 #  frequency in Hz
omega2 = 2*pi*f2 #  angular frequency in Hz
epsilon_r = 81 #  relative permittivity
epsilon_0 = 8.854*10**(-12) #  permittivity in free space
epsilon = epsilon_r*epsilon_0 #  permittivity
mu_r = 1 #  relative permeability
mu_0 = 4*pi*10**(-7) #  permeability in free space
mu = mu_r*mu_0 #  permeability

#calculations and results

print "when frequency = 25kHz"
alpha_1 = omega1*sqrt((mu*epsilon)/2*(sqrt(1+(sigma**2/(omega1**2*epsilon**2)))-1)) #  attenuation constant when f = 25kHz
# formula: exp(-alpha*x) = 0.1
x1 = 2.3/alpha_1 #  transmitted distance in m
print "transmitted distance in m:",round(x1,3)
print "when frequency = 25MHz"
alpha_2 = omega2*sqrt((mu*epsilon)/2*(sqrt(1+(sigma**2/(omega2**2*epsilon**2)))-1)) #  attenuation constant when f = 25MHz
x2 = 2.3/alpha_2 #  transmitted distance in m
print "transmitted distance in m:",round(x2,3)


# note: the values of epsilon_r = 81 and of mu_r = 1 for sea water which are not given in the book.

from __future__ import division
from math import sqrt,pi,radians,asin,cos,sin,degrees

#variable Declaration

E_i = 1  #  magnitude of incident electric field in mV/m
E_i = 1*10**-3  #  magnitude of incident electric field in V/m
epsilon_0 = 8.854*10**-12  #  permittivity in free space in F/m
mu_0 = 4*pi*10**-7  #   permeability in free space in H/m
theta_i = 15  #  incident angle in degrees
epsilon_r1 = 8.5  #  relative permittivity of medium 1
mu_r1 = 1  #  relative permeability of medium 1
epsilon1 = epsilon_r1*epsilon_0  #  permittivity
mu1 = mu_0*mu_r1  #  permeability
eta1 = sqrt(mu1/epsilon1)  #  intrinsic impedence of medium 1 in ohm
epsilon2 = epsilon_0  #  permittivity of medium 2
mu2 = mu_0  #  permeability of medium 2
eta2 = sqrt(mu2/epsilon2)  #  intrinsic impedence of medium 2 in ohm

#calculations and result

# formula : sin(theta_i)/sin(theta_t) = sqrt(epsilon2/epsilon1)
theta_t = asin(sin(radians(theta_i)))/(sqrt(epsilon2/epsilon1))  #  transmitted angle in degrees
E_r = (E_i*(((eta2*cos(radians(theta_i))))-(eta1*cos(radians((theta_i))))))/((eta2*cos(radians(theta_i)))+(eta1*cos(radians(theta_i))))  #  reflection cofficient of electric field
print "reflection cofficient of electric field in mV/m:",round(E_r*1000,3)
H_i = E_i/eta1  #  incident cofficient of magnetic field
print "incident cofficient of magnetic field in micro*A/m:",round(H_i*10**6,3)
H_r = E_r/eta1  #  reflection cofficient of electric field
print "reflection cofficient of magnetic field in micro*A/m:",round(H_r*10**6,3)


#note : minute difference in decimel in the value of H_i and H_r.

from __future__ import division
from math import pi,sqrt

#variable declaration

sigma = 5.8*10**7 #  conductivity in mho/m
f = 2 #  frequency in MHz
f = 2*10**6 #  frequency in Hz
omega = 2*pi*f #  angular frequency in rad/sec
E = 2 #  magnitude of electric field in mV/m
E = 2*10**-3 #  magnitude of electric field in V/m
epsilon_0 = 8.854*10**-12 #  permittivity in free space in F/m
mu_0 = 4*pi*10**-7 #   permeability in free space in H/m
epsilon_r = 1 #  relative permittivity
mu_r = 1 #  relative permeability
epsilon = epsilon_r*epsilon_0 #  permittivity
mu = mu_0*mu_r #  permeability

# calculations
eta = sqrt(mu*omega/sigma) #  intrinsic impedence in ohm
P_av = (1/2)*E**2/eta #  average power density anbsorbed by copper

#result
print "average power density anbsorbed by copper in mW/m**2:",round(P_av*1000,2)

