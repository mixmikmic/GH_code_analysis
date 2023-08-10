import matplotlib.pyplot as plt
from numpy import sqrt,pi,arange,cos,sin
from qutip import *

get_ipython().magic('matplotlib inline')

pz = Qobj([[1],[0]])
mz = Qobj([[0],[1]])
px = Qobj([[1/sqrt(2)],[1/sqrt(2)]])
mx = Qobj([[1/sqrt(2)],[-1/sqrt(2)]])
py = Qobj([[1/sqrt(2)],[1j/sqrt(2)]])
my = Qobj([[1/sqrt(2)],[-1j/sqrt(2)]])
Sx = 1/2.0*sigmax()
Sy = 1/2.0*sigmay()
Sz = 1/2.0*sigmaz()

omega = 2*pi
H = -omega*Sz

t = arange(0,4*pi/omega,0.05)

result1 = sesolve(H, px, t, [Sx,Sy,Sz])

for r in result1.expect:
    plt.plot(result1.times*omega/pi, r)

result2 = sesolve(H, pz, t, [Sx,Sy,Sz])

for r in result2.expect:
    plt.plot(result2.times*omega/pi, r)

psi = 1/sqrt(2)*tensor(pz, mz) + 1/sqrt(2)*tensor(mz, pz)
#psi = tensor(mz,pz)

omega = 5
H = -omega*tensor(Sz,Sz)

result3 = sesolve(H, psi, t, tensor(pz,mz)*tensor(pz,mz).dag())

plt.plot(result3.times,result3.expect[0])

omega=2
H = -omega/sqrt(2)*(Sz + Sx)
t = arange(0,2*pi/omega,0.05)

result4 = sesolve(H, px, t, [Sx, Sy, Sz])

for r in result4.expect:
    plt.plot(result4.times*omega/pi, r)

sx, sy, sz = result4.expect

b = Bloch()
b.add_points([sx,sy,sz])
b.zlabel = ['$\\left|+z\\right>$', '$\\left|-z\\right>$']
b.view = [-45,20]
b.show()

omega0 = 2.0 * 2 * pi   # pick a nice value for a frequency, note this is 1 Hz
omega1 = 0.25 * 2 * pi  # 25% of omega0
w = 2.0 * 2 * pi        # the driving frequency

H0 = - omega0 * Sz      # the first term in H
H1 = - omega1 * Sx      # the second term in H

omegaR = sqrt((w - omega0)**2 + (omega1/2.0)**2)
t = arange(0,3.0 * 2 * pi / omegaR,0.02)   # scale the time by omegaR, plot 3 units of 2pi/omegaR

#args = [H0, H1, w]      # parts of the Hamiltonian 

def H1_coeff(t, args):
    return cos(w * t)

H = [H0, [H1, H1_coeff]]

result5 = sesolve(H, pz, t, [Sx, Sy, Sz, mz*mz.dag()],args)
sx, sy, sz, Pz = result5.expect

b2 = Bloch()
b2.add_points([sx,sy,sz])
b2.show()

plt.plot(result5.times*omegaR/pi,Pz)

omega0 = 1.0 * 2 * pi   # pick a nice value for a frequency, note this is 1 Hz
omega1 = 0.05 * 2 * pi  # 25% of omega0
w = 1.0 * 2 * pi        # the driving frequency

H0 = - omega0 * Sz      # the first term in H
H1 = - omega1 * Sx      # the second term in H

omegaR = sqrt((w - omega0)**2 + (omega1/2.0)**2)
t = arange(0,3.0 * 2 * pi / omegaR,0.05)   # scale the time by omegaR, plot 3 units of 2pi/omegaR

def H1_coeff2(t, args):       # this function calculates H at each time step t
    if t < 2*pi/omegaR * 0.5:  # only add the H1 piece for the first chunk of time.
        coeff = cos(w * t)
    else:
        coeff = 0
    return coeff

H = [H0, [H1, H1_coeff2]]

result6 = sesolve(H, pz, t, [Sx, Sy, Sz, mz*mz.dag()],args)
sx, sy, sz, Pz = result6.expect

plt.plot(result6.times,Pz)
plt.ylim(-0.1,1.1)
plt.xlim(-5,125)



