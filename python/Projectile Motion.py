get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy import *
from scipy.integrate import odeint, ode

# set constants and define differentials

g = 9.82 # m/s^2
m = 0.15 # kg
D = 0.07 # m
gamma = 0.25 # N*s^2/m^4
c = gamma*D**2 

def dx(x, t):
    """
    The right-hand side of the coupled equations
    """
    x, y, Dx, Dy = x[0], x[1], x[2], x[3]
    
    dx1 = Dx
    dx2 = Dy
    dx3 = -(c/m)*(Dx**2 + Dy**2)**(1./2)*Dx
    dx4 = -g -(c/m)*(Dx**2 + Dy**2)**(1./2)*Dy
    
    return [dx1, dx2, dx3, dx4]

# choose an initial state

initial_velocity = 30  # m/s
initial_angle = 50     # degrees

v0x = initial_velocity*cos(float(initial_angle)/180*pi)
v0y = initial_velocity*sin(float(initial_angle)/180*pi)

x0 = [0, 0, v0x, v0y]  # (initial angle, initial angular velocity)
print v0x, v0y

# time coodinate to solve the ODE for: from 0 to 10 seconds

t = linspace(0, 10, 501)

# solve the coupled ODE 

x = odeint(dx, x0, t)  # note that the matrix notation takes care of the individual differentials

# x = [x-coord(t), y-coord(t), x-vel(t), y-vel(t)]

x.shape 

# projectile motion without resistance

xx = 0 + x0[2]*t
yy = 0 + x0[3]*t - g/2*t**2

# locate indicies for integral times t = 1,2,3,4,5,6,7,8

idt = []
for k in range(1,9):
    tt = abs(t - k) < 10**(-4)
    print where(tt)[0][0], k
    idt.append(where(tt)[0][0])

# plot the projectiles as a function of time and space

f, ax = plt.subplots(facecolor='white',figsize=(14,10))
ax.plot(xx, yy, '--b', label="projectile without wind resistance")
ax.plot(x[:, 0], x[:, 1], '-r', label="projectile with wind resistance")
ax.plot(x[idt, 0], x[idt, 1], 'ok', label="integral times")
ax.plot(xx[idt], yy[idt], 'ok')
ax.set_title('Projectile Motion')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_ylim(-150,35)
ax.set_yticks(linspace(-150,30,37))
ax.grid()
ax.legend()
ax.annotate('t = 8s', xy=(85, -107), xytext=(65, -107),
            arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth = 3),
            )
ax.annotate('t = 8s', xy=(158, -130), xytext=(170, -130),
            arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth = 3),
            )

