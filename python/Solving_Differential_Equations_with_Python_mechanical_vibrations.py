# setup 
from sympy import Eq, pi
import sympy as sp
import matplotlib.pyplot as plt
from numpy import linspace
sp.init_printing(use_latex='mathjax')
get_ipython().magic('matplotlib inline') # inline plotting

t,k,m,c = sp.symbols('t,k,m,c')
x = sp.Function('x') # (t)

k_ = 1e3  # spring constant, kN/m
m_ = 50 # mass, Kg
c_ = 3  # damping coefficient 

ode = k*x(t) + m*c*x(t).diff(t,1) + m*x(t).diff(t,2)
Eq(ode)

ode_sol = sp.dsolve(ode)
ode_sol

x0 = 1.0
v0 = 0

bc1 = Eq(ode_sol.lhs.subs(x(t),x0), ode_sol.rhs.subs(t,0))
bc2 = Eq(ode_sol.lhs.subs(x(t),v0), ode_sol.rhs.diff(t).subs(t,0))

C_eq = {bc1,bc2}
C_eq

known_params = {m,c,k,t}
const = ode_sol.free_symbols - known_params
const

Csol = sp.solve(C_eq,const)
Csol

ode_sol = ode_sol.subs(Csol)
ode_sol

ode_sol = ode_sol.subs({m:m_, c:c_, k:k_})
ode_sol

#sp.plot(ode_sol.rhs, (t,0,5)) ;

xfun = sp.lambdify(t,ode_sol.rhs, "numpy")
vfun = sp.lambdify(t,sp.diff(ode_sol.rhs), "numpy")

t = linspace(0,5,1000)

fig, ax1 = plt.subplots(figsize=(12,8))
ax2 = ax1.twinx()
ax1.plot(t,xfun(t),'b',label = r'$x (mm)$', linewidth=2.0)
ax2.plot(t,vfun(t),'g--',label = r'$\dot{x} (m/sec)$', linewidth=2.0)
ax2.legend(loc='lower right')
ax1.legend()
ax1.set_xlabel('time , sec')
ax1.set_ylabel('disp (mm)',color='b')
ax2.set_ylabel('velocity (m/s)',color='g')
plt.title('Mass-Spring System with $v_0=0.1%f' % (v0))
plt.grid()
plt.show()

from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,xlabel,ylabel,title,legend,figure,subplots
from numpy import cos, pi, arange, sqrt, pi, array
get_ipython().magic('matplotlib inline') # inline plotting

def MassSpringDamper(state,t):
    '''
    k=spring constant, Newtons per metre
    m=mass, Kilograms
    c=dampign coefficient, Newton*second / meter    

    for a mass, spring, damper 
        xdd = -k*x/m -c*xd
    '''
  
    k=1e3  # spring constant, kN/m
    m=50 # mass, Kg
    c=3  # damping coefficient 
    # unpack the state vector
    x,xd = state # displacement,x and velocity x'
    # compute acceleration xdd = x''
    xdd = -k*x/m -c*xd
    return [xd, xdd]

x0 = 1.0
v0 = 0
state0 = [x0, v0]  #initial conditions [x0 , v0]  [m, m/sec] 
ti = 0.0  # initial time
tf = 5.0  # final time
step = 0.001  # step
t = arange(ti, tf, step)
state = odeint(MassSpringDamper, state0, t)
x = array(state[:,[0]])
xd = array(state[:,[1]])

# Plotting displacement and velocity
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t,x,'b',label = r'$x (mm)$', linewidth=2.0)
ax2.plot(t,xd,'g--',label = r'$\dot{x} (m/sec)$', linewidth=2.0)
ax2.legend(loc='lower right')
ax1.legend()
ax1.set_xlabel('time , sec')
ax1.set_ylabel('disp (mm)',color='b')
ax2.set_ylabel('velocity (m/s)',color='g')
plt.title('Mass-Spring System with $v_0=%0.1f$' % (v0))
plt.grid()
plt.show()



