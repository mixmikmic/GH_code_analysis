from IPython.display import SVG
SVG(filename='coupling.svg')

import csv
import numpy as np
    
# Drag coefficient look up
C_d_t = []
Ma_t = []
f = open('CD_sustainer_poweron.csv') # Use aerobee 150 drag data
aerobee_cd_data = csv.reader(f, delimiter=',')
for row in aerobee_cd_data:
    C_d_t.append(row[1])
    Ma_t.append(row[0])

# Plot the results
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
get_ipython().run_line_magic('config', "InlineBackend.figure_formats=['svg']")
get_ipython().run_line_magic('matplotlib', 'inline')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

Ma_t = np.array(Ma_t, dtype=float)
C_d_t = np.array(C_d_t, dtype=float)
plt.plot(Ma_t, C_d_t)
plt.xlabel('Mach number')
plt.ylabel('Cd')
plt.show()

# Airframe Section Parameters
from math import pi
m  = 0.463              # Single LV3 airframe module mass [kg], 18 in.
Do = 0.168              # Outer diameter                  [m]
L  = 0.457              # Length                          [m]
A_fiber = pi * Do *L
rho_fiber = m/A_fiber   # Planar density                  [kg/m^2]
print("Module planar density = {0:.2f} kg/m^2".format(rho_fiber))

from math import sqrt, pi, exp, log, cos
import numpy as np


# A simple forward Euler integration for rocket trajectories
def dry_mass(L, dia):
    m_avionics = 3.3                                                         # Avionics mass [kg]
    m_recovery = 4                                                           # Recovery system mass [kg]
    m_payload = 1 + 3                                                        # Payload mass [kg]
    m_tank_fudgefactor = 1.5
    m_tankage = rho_fiber * m_tank_fudgefactor * L * dia * pi + 3                 # Tank mass Estimation [kg]
    m_engine = 3                                                             # Engine mass [kg]
    m_feedsys = 10                                                           # Feed system mass [kg]
    len_airframe = 1                                                         # Length of Airframe (sans tankage) [m]
    m_airframe  = rho_fiber * m_tank_fudgefactor * len_airframe * dia * pi + 1    # Airframe mass [kg]
    return (m_avionics + m_recovery + m_payload + m_tankage 
    + m_engine + m_feedsys + m_airframe)                                     # Total dry mass [kg]
    
def propellant_mass(A, L, OF=1.3):
    rho_alc = 852.3             # Density, ethanol fuel [kg/m^3]
    rho_lox = 1141.0            # Density, lox          [kg/m^3]
    L_lox = L/(rho_lox/(rho_alc*OF) + 1)
    m_lox = rho_lox*L_lox*A     # Oxidizer mass         [kg]
    m_alc = rho_alc*(L-L_lox)*A # Fuel mass             [kg]
    return m_alc + m_lox        # Propellant Mass       [kg]

def std_at(h):                  # U.S. 1976 Standard Atmosphere
    if h < 11000:
        T = 15.04 - 0.00649*h
        p = 101.29*((T + 273.1)/288.08)**5.256

    elif 11000 <= h and h <25000:
        T = -56.46
        p = 22.65*exp(1.73 - 0.000157*h)

    else:
        T = -131.21 + 0.00299*h
        p = 2.488 * ((T + 273.1)/216.6)**(-11.388)

    rho = p/(0.2869*(T + 273.1)) # Ambient air density [kg/m^3]
    p_a = p*1000                 # Ambient air pressure [Pa]
    T_a = T + 273.1              # Ambient air temperature [K]
    return p_a, rho, T_a

def thrust(x, p_ch, T_ch, p_e, ke, Re, mdot):
    p_a = std_at(x)[0]                                          # Ambient air pressure [Pa]
    p_t = p_ch * (1 + (ke - 1)/2)**(-ke/(ke - 1))               # Throat pressure      [Pa]
    T_t = T_ch * (1/(1 + (ke - 1)/2))                           # Throat temperature   [K]
    A_t = (mdot / p_t) * sqrt(Re * T_t/ke)                      # Throat area          [m^2]
    A_e = A_t * (2/(ke + 1))**(1/(ke - 1)) * (p_ch/p_e)**(1/ke) *     1/sqrt((ke + 1)/(ke - 1) * (1 - (p_e/p_ch)**((ke - 1)/ke)))    # Exit area [m^2]
    
    ex = A_e/A_t                                                 # Expansion ratio
    #alpha_t = [14, 11, 10, 9]                                    # Lookup table of divergence angles, assuming 80% bell length
    #ex_t = [5, 10, 15, 20]                                       # Lookup table of expansion ratios from alpha_t
    #alpha= np.interp(ex, ex_t, alpha_t)
    #lam = 0.5 * (1 + cos(alpha * pi/180))                        # Thrust cosine loss correction (not an O(1) effect, even in extreme cases) 
    lam = 1
    Ve = lam * sqrt(2 * ke/(ke - 1) * Re * T_ch *     (1 - (p_e/p_ch)**((ke - 1)/ke))) * 0.9                       # Exhaust velocity    [m/s]
    
    F = mdot * Ve + (p_e - p_a) * A_e                            # Thrust force, ignoring that isp increases w/ p_ch [N]
    return F, A_t, A_e, Ve

def drag(x, v, A, Ma, C_d_t, Ma_t):
    (p_a, rho, T_a) = std_at(x)
    #C_d_t = [0.15, 0.15, 0.3, 0.45, 0.25, 0.2, 0.175, .15, .15] # V2 rocket drag coefficient lookup table
    #Ma_t = [0, 0.6, 1.0, 1.1, 2, 3, 4, 5, 5.6]                  # V2 rocket Mach number lookup table
    C_d = np.interp(Ma, Ma_t, C_d_t)                            # Drag coefficient function
    q = 0.5 * rho * v**2                                        # Dyanmic pressure [Pa]
    D = q * C_d * A                                             # Drag force       [N]
    return D, q

def trajectory(L, mdot, dia, p_e, p_ch=350, T_ch=3500, ke=1.3, Re=349, x_init=0):
    # Note combustion gas properties ke, Re, T_ch, etc, determined from CEA
    # Physical constants
    g_0 = 9.81 # Gravitational acceleration [m/s^2]
    dt = 1     # Time step                  [s]
    ka = 1.4   # Ratio of specific heats, air  
    Ra = 287.1 # Avg. specific gas constant (dry air)
    
    # LV4 design variables
    dia = dia*0.0254       # Convert in. to m
    A = pi*(dia/2)**2      # Airframe frontal area projected onto a circle of diameter variable dia
    m_dry = dry_mass(L, A) # Dry mass, call from function dry_mass()
    mdot = mdot            # Mass flow rate [kg/s]
    p_ch = p_ch*6894.76    # Chamber pressure, convert psi to Pa
    p_e = p_e*1000         # Exit pressure, convert kPa to Pa

    # Initial conditions
    x = [x_init]
    v = [0.]
    a = [0.]
    t = [0.]
    rho = [std_at(x[-1])[1]]
    p_a = [std_at(x[-1])[0]]
    T_a = [std_at(x[-1])[2]]
    m_prop = [propellant_mass(A, L)]
    m = [m_dry + m_prop[-1]]
    (F, A_t, A_e, Ve) = thrust(x[-1], p_ch, T_ch, p_e, ke, Re, mdot)
    F = [F]
    D = [0.]
    Ma = [0.]
    q = [0.]
    r = (m_prop[0] + m_dry)/m_dry # Mass ratio
    dV1 = Ve*log(r)/1000          # Tsiolkovsky's bane (delta-V)

    while True:
        p_a.append(std_at(x[-1])[0])
        rho.append(std_at(x[-1])[1])
        T_a.append(std_at(x[-1])[2])
        # Check of the propellant tanks are empty
        if m_prop[-1] > 0:
            (Fr, A_t, A_e, Ve) = thrust(x[-1], p_ch, T_ch, p_e, ke, Re, mdot)
            F.append(Fr)
            m_prop.append(m_prop[-1] - mdot*dt)
            mdot_old = mdot
        else:
            Ve = thrust(x[-1], p_ch, T_ch, p_e, ke, Re, mdot_old)[3]
            F.append(0)
            mdot = 0
            m_prop[-1] = 0
        q.append(drag(x[-1], v[-1], A, Ma[-1], C_d_t, Ma_t)[1])
        D.append(drag(x[-1], v[-1], A, Ma[-1], C_d_t, Ma_t)[0])
        a.append((F[-1] - D[-1])/m[-1] - g_0)
        v.append(a[-1]*dt + v[-1])
        x.append(v[-1]*dt + x[-1]) 
        Ma.append(v[-1]/sqrt(ka*Ra*T_a[-1]))
        t.append(t[-1] + dt)
        m.append(m_dry + m_prop[-1])
        TWR = a[1]/g_0      # Thrust-to-weight ratio constraint
        ex = A_e/A_t
        S_crit = p_e/p_a[0] # Sommerfield criterion constraint
        if v[-1] <= 0:
            x = np.array(x)
            a = np.array(a)
            F = np.array(F)
            D = np.array(D)
            q = np.array(q)
            return x, v, a, t, F, D, Ma, rho, p_a, T_a, TWR, ex, Ve, A_t, dV1, m, S_crit, q, m_prop

# Pseudo-objective function
global cons_TWR, cons_S_crit, cons_accel, cons_LD, cons_alt

#CHANGE CONSTRAINTS HERE
cons_TWR = 8       # TWR constraint
cons_S_crit = 0.35 # Critical pressure ratio constraint
cons_accel = 15    # Max acceleration constraint
cons_LD = 15       # L/D ratio constraint
cons_alt = 100000  # Min altitude constraint

def f(x, p_ch=350, rp=50): ##CHANGE CHAMBER PRESSURE HERE
    L = x[0]    # Rocket length (m)
    mdot = x[1] # Propellant mass flow rate (kg/s)
    dia = x[2]  # Rocket diameter (in)
    p_e = x[3]  # Pressure (kPa)
    (alt, v, a, t, F, D, Ma, rho, p_a, T_a, TWR, ex, Ve, A_t, dV1, m, S_crit, q, m_prop)     = trajectory(L, mdot, dia, p_e, p_ch)


    obj_func = m[0]/m0 + rp*(max(0, (L+2)/(dia*0.0254*cons_LD) - 1)**2 +                max(0, -TWR/cons_TWR + 1)**2 + max(0, -S_crit/cons_S_crit + 1)**2 +                              max(0, -alt[-1]/cons_alt + 1)**2 + max(0, max(abs(a))/(cons_accel*9.81) - 1)**2)
    return obj_func

##CHANGE INITIAL DESIGN GUESS HERE
L = 1.5    # Rocket length (m)
mdot = 0.453592 * 0.9 * 5 # Propellant mass flow rate (kg/s)
dia = 9  # Rocket diameter (in)
p_e = 40  # Pressure (kPa)
X0 = np.array([L, mdot, dia, p_e])
m0 = trajectory(X0[0], X0[1], X0[2], X0[3])[-4][0] # Initial guess GLOW

from scipy.optimize import minimize
res = minimize(f, X0, method='nelder-mead')  

# Plot the results
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pylab
get_ipython().run_line_magic('config', "InlineBackend.figure_formats=['svg']")
get_ipython().run_line_magic('matplotlib', 'inline')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

pylab.rcParams['figure.figsize'] = (10.0, 10.0)

def plot_trajectory(res):
    p_ch = 350 # Chamber pressure [psi] **DONT FORGET TO CHANGE THE VALUE IN THE OBJECTIVE FUNCTION IN def f()**
    (alt, v, a, t, F, D, Ma, rho, p_a, T_a, TWR, ex, Ve, A_t, dV1, m, S_crit, q, m_prop) = trajectory(res.x[0], res.x[1], res.x[2], res.x[3], p_ch)
    # Rename the optimized output
    L = res.x[0]    
    mdot = res.x[1]
    dia = res.x[2]
    p_e = res.x[3]

    f, (ax1, ax2, ax3, ax4, ax6, ax7, ax8) = plt.subplots(7, sharex=True)
    for n in (ax1, ax2, ax3, ax4, ax6, ax7, ax8):
        n.spines['top'].set_visible(False)
        n.spines['right'].set_visible(False)
        n.yaxis.set_ticks_position('left')
        n.xaxis.set_ticks_position('bottom')
        n.yaxis.labelpad = 20
        
    ax1.plot(t, alt/1000, 'k')
    ax1.set_ylabel("Altitude (km)")
    ax1.yaxis.major.locator.set_params(nbins=6)
    #ax1.set_title('LV4 Trajectory')
    ax2.plot(t, v, 'k')
    ax2.yaxis.major.locator.set_params(nbins=6)
    ax2.set_ylabel("Velocity (m/s)")
    ax3.plot(t, a/9.81, 'k')
    ax3.yaxis.major.locator.set_params(nbins=10)
    ax3.set_ylabel("Acceleration/g0")
    ax4.plot(t, F/1000, 'k')
    ax4.yaxis.major.locator.set_params(nbins=6)
    ax4.set_ylabel("Thrust (kN)")
    ax6.plot(t, q/1000, 'k')
    ax6.yaxis.major.locator.set_params(nbins=6)
    ax6.set_ylabel("Dynamic Pressure (kPa)")
    ax7.plot(t, Ma, 'k')
    ax7.yaxis.major.locator.set_params(nbins=6) 
    ax7.set_ylabel("Mach number")
    ax7.set_xlabel("t (s)")
    ax8.plot(t, np.array(m)*0.666*np.array(a), 'k')
    #ax8.yaxis.major.locator.set_params(nbins=6) 
    ax8.set_ylabel("LOX Tank Axial Load")
    ax8.set_xlabel("t (s)")
    plt.savefig(traj.svg)
    plt.show()
    
    
    np.set_printoptions(precision=3)
    print('\n')
    print('OPTIMIZED DESIGN VECTOR')
    print('-----------------------------')
    print('x_optimized                                = ', res.x)
    print('x_initial_guess                            = ', X0)
    print('design tankage length                      = {0:.2f} m'.format(res.x[0]))
    print('design mass flow rate                      = {0:.2f} kg/s'.format(res.x[1]))
    print('design airframe diameter                   = {0:.2f} in.'.format(res.x[2]))
    print('design nozzle exit pressure                = {0:.2f} kPa'.format(res.x[3]))
    print('iterations                                 =', res.nit)
    print('design GLOW                                = {0:.1f} kg'.format(m[0]))
    print('x0 GLOW                                    = {0:.1f} kg'.format(           trajectory(X0[0], X0[1], X0[2], X0[3], p_ch)[-4][0]))

    print('\n')
    print('CONSTRAINTS')
    print('-----------------------------')
    print('L/D ratio (c.f. < {})                      = {:.2f}'.format(cons_LD, (L+2)/(dia*0.0254)))
    print('Sommerfield criterion (c.f. pe/pa >= {})   = {:.1f}'.format(cons_S_crit, S_crit))
    print("Max acceleration (c.f. < {})               = {:.2f} g's".format(cons_accel, max(abs(a))/9.81))
    print('TWR at lift off (c.f. > {})                = {:.2f}'.format(cons_TWR, TWR))
    print('altitude at apogee                         = {0:.1f} km'.format(alt[-1]/1000))

    print('\n')
    print('ADDITIONAL INFORMATION')
    print('-----------------------------')
    print('mission time at apogee                     = {0:.1f} s'.format(t[-1]))
    print('design total propellant mass               = {0:.3f} kg'.format(m_prop[0]))
    print('design thrust (sea level)                  = {0:.1f} kN'.format(F[0]/1000))
    j = 0
    for thing in F:
        if thing == 0:
            fdex = j
            break
        j += 1
    print('design thrust (vacuum)                     = {0:.1f} kN'.format(F[fdex - 1]/1000))
    print('design burn time                           = {} s'.format(fdex))
    print('design expansion ratio                     = {0:.1f}'.format(ex))
    print('design throat area                         = {0:.1f} in.^2'.format(A_t/0.0254**2))
    print('design isp                                 = {0:.1f} s'.format(Ve/9.81))
    print('design chamber pressure                    = {0:.1f} psi'.format(p_ch))
    print('design total impulse                       = {0:.1f} kN*s'.format(j*(F[fdex - 1]/1000 + F[0]/1000)/2))
    print('design dV                                  = {0:.1f} km/s'.format(dV1))
    print('estimated minimum required dV              = {0:.1f} km/s'.format( sqrt(2*9.81*alt[-1])/1000))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

plot_trajectory(res)

# Fine Tuning
#res.x = [   1.585,    2.58,     8.987,  60]

#res.x =  [   1.259,    3.11,    10.847,  150.025] #Is an especially good design vector.
#plot_trajectory(res)

# Class simplex: 
# Nelder-Mead simplex search
import numpy as np
from math import sqrt, pi, exp, log, cos
import math as m

def search(f, x_start, max_iter = 100, gamma = 5, beta = 0.5, rp=100, a=10, epsilon = 1E-6):
    """
    parameters of the function:
    f is the function to be optimized
    x_start (numpy array) is the initial simplex vertices
    epsilon is the termination criteria
    gamma is the contraction coefficient
    beta is the expansion coefficient
    """
    # Init Arrays
    N = len(x_start)     # Amount of design variables
    fb = []              # Empty function matrix
    xnew = []            # Empty re-write for design variables
    x    = []            # Empty x matrix
    C    = [[0]*N]*(N+1) # Empty center point matrix #####CHANGED
    
    # Generate vertices of initial simplex
    x0 = (x_start)   # x0 Value for x Matrix
    x1 = [x0 + [((N + 1)**0.5 + N - 1.)/(N + 1.)*a, 0., 0., 0.]]
    x2 = [x0 + [0., ((N + 1)**0.5 - 1.)/(N + 1.)*a, 0., 0.]]
    x3 = [x0 + [0., 0., ((N + 1)**0.5 - 1.)/(N + 1.)*a, 0.]]
    x4 = [x0 + [0., 0., 0., ((N + 1)**0.5 - 1.)/(N + 1.)*a]]
    x = np.vstack((x0, x1, x2, x3, x4))

    # Simplex iteration
    while True:
        # Find best, worst, 2nd worst, and new center point
        f_run = np.array([f(x[0], rp), f(x[1], rp), f(x[2], rp), f(x[3], rp), f(x[4], rp)]).tolist()         # Func. values at vertices
        xw = x[f_run.index(sorted(f_run)[-1])] # Worst point
        xb = x[f_run.index(sorted(f_run)[0])]  # Best point
        xs = x[f_run.index(sorted(f_run)[-2])] # 2nd worst point        
        for i in range(0, N+1):
            if i == f_run.index(sorted(f_run)[-1]):
                C[i] = [0,0,0,0]
            else:
                C[i] = x[i].tolist()
        xc = sum(np.array(C))/(N) # Center point
        xr = 2*xc - xw            # Reflection point
        fxr = f(xr, rp)
        fxc = f(xc, rp)
        
        # Check cases
        # f(xr, rp) < f(xb, rp): # Expansion
        if fxr < f_run[f_run.index(sorted(f_run)[0])]:    
            xnew = (1 + gamma)*xc - gamma*xr
        # f(xr, rp) > f(xw, rp): # Contraction 1
        elif fxr > f_run[f_run.index(sorted(f_run)[-1])]:
            xnew = (1 - beta)*xc + beta*xw
        # f(xs, rp) < f(xr, rp) and f(xr, rp) < f(xw, rp): # Contraction 2
        elif f_run[f_run.index(sorted(f_run)[-2])] < fxr and fxr < f_run[f_run.index(sorted(f_run)[-1])]: 
            xnew = (1 + beta)*xc - beta*xw
        else:
            xnew = xr
        
        # Replace Vertices
        x[f_run.index(sorted(f_run)[-1])] = xnew
        #x[f_run.index(sorted(f_run)[1])] = xb # Replace best
        #x[f_run.index(sorted(f_run)[2])] = xs # Replace second best
        fb.append(f(xb, rp))
        print('Current optimum = ', fb[-1])
        
        # Break if any termination critera is satisfied
        if len(fb) == max_iter: #or term_check(x, xc, xw, N, rp, f_run) <= epsilon:
            (alt, v, a, t, F, D, Ma, rho, p_a, T_a, TWR, ex, Ve, A_t, dV1, m, S_crit, q, m_prop, p_ch)             = trajectory(xb[0], xb[1], xb[2], xb[3])
            return f(x[f_run.index(sorted(f_run)[0])], rp), x[f_run.index(sorted(f_run)[0])], len(fb)
        
def term_check(N, rp, f_run, fxc): # Termination critera
    M = [0]*(N + 1)
    for i in range(0, N + 1):
        if i == f_run.index(sorted(f_run)[-1]): # Avoid worst point
            M[i] = 0
        else:
            M[i] = (f_run[i] - fxc)**2
    return m.sqrt(sum(M)/N)



