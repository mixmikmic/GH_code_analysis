import math

rho=1.12; # [kg/m3], at sea level 1.225, 1000m 1.121
g=9.81;

m_rocket = 23.34 # mass after burnout [kg]
m_glider = 0.28 # [kg]

def F_drag(v, cd, d):
    A = d**2*math.pi/4
    return 1/2*rho*v**2*cd*A

def v_descent(m, cd, d):
    A = d**2*math.pi/4
    return math.sqrt(2*m*g/(rho*cd*A))

cd_drogue = 1.6 # drag coefficient
d_drogue = 0.46 # diameter [m]
v_drogue_init = v_descent(m_rocket, cd_drogue, d_drogue)
v_drogue = v_descent(m_rocket - m_glider, cd_drogue, d_drogue)
print('{:.2f} m/s ({:.0f} km/h)'.format(v_drogue_init, v_drogue_init*3.6))
print('{:.2f} m/s ({:.0f} km/h) (after glider deployment)'      .format(v_drogue, v_drogue*3.6))

cd_main = 2.2 # drag coefficient
d_main = 2.13 # diameter [m]
v_main = v_descent(18, cd_main, d_main)
print('18kg: {:.2f} m/s ({:.0f} km/h)'.format(v_main, v_main*3.6))

cx = 1.8

# deceleration shock from drogue deployment
v_apogee = 250/3.6 # [m/s] worst case lateral velocity at apogee
shock_drogue = cx * F_drag(v_apogee, cd_drogue, d_drogue)
print('drogue: {:.0f} N'.format(shock_drogue))

# deceleration shock from drogue to main chute descent velocity
shock_main = cx * F_drag(v_drogue, cd_main, d_main)
print('main: {:.0f} N'.format(shock_main))

A_main = d_main**2*math.pi/4
M = 25 # kg
mass_ratio = rho * (A_main * cd_main)**1.5/M
print('mass ratio for main parachute and 25kg: {:.3}'      .format(mass_ratio))

cx_finite_mass = 0.2
shock_main_finite_mass_model = cx_finite_mass * F_drag(v_drogue, cd_main, d_main)
print('main shock with finite mass model (25kg): {:.0f} N'      .format(shock_main_finite_mass_model))

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

mp = 0.75 # [kg]
mr = m_rocket-m_glider # [kg]

k_l = 44162 # [N], measured
l = 5.95 # [m] shock cord length
k = k_l/l # [N/m]

print('k = {} [N/m]'.format(k))

def F_shockcord(yr, yp):
    dl = (yp-yr)
    if dl < 0:
        return 0
    else:
        return k*dl
    
def F_drag_sign(v):
    f = F_drag(v, cd_main, d_main)
    return math.copysign(f, -v)

y0 = [0, -v_drogue, 0, -v_drogue]
def dydt(y, t):
    vr = y[1]
    Fs = F_shockcord(y[0], y[2])
    ar = 1/mr*Fs - g
    vp = y[3]
    Fd = F_drag_sign(y[3])
    ap = 1/mp*Fd - 1/mp*Fs - g
    return [vr,ar,vp,ap]

t = np.linspace(0,3,1000)
sol = odeint(dydt, y0, t)

fn = np.vectorize(lambda n: F_shockcord(sol[n][0], sol[n][2]))
fs = fn(np.array(range(len(t))))
plt.plot(t,fs, label='F_shockcord')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('[N]')
plt.grid()
plt.show()

print('shock force: {:.0f} [N]'.format(max(fs)))




