import math as m
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import pylab
from matplotlib import rc
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats=['svg']")

# Ondrej Fercak
# Ian Zabel
# Note: - Dimensions from LV4 MDO
#       - Atmo pressure from LV4 MDO
#       - All calculations assume hemisphere endcaps

# Dimensions and Material Properties

Do = 0.254  # Outer tank diameter (m)
Ro = Do/2   # Outer tank radius (m)
L  = 1.2    # Overall tank length (m)
TW = 15     # Thin-walled assumption limit
t  = 0.002  # Wall thickness (m)
FS = 1.1    # Factor of safety
Sy = 240e6  # Yield strength, Al (Pa)
Su = 290e6  # Ultimate tensile strength, Al (Pa)
TL = Ro*0.3 # Transition region length (m)

# Propellant Parameters
# Note: Volume values TEMP

OF   = 1.3   # Oxidizer to fuel ratio
rhof = 852.3 # Density, ethanol fuel (kg/m^3)
rhoo = 1141  # Density, lox (kg/m^3)

# Pressure
# Note: Use either np.all() or np.any() for ineq. def

def atmo(x):
    if np.all(x < 11000):
        T = 15.04-0.0065*x
        P = 101.3*((T+273.1)/288.1)**5.26
    elif np.all(11000 <= x) & np.all(x < 25000):
        T = -56.46
        P = 22.65*exp(1.73-0.00016*x) 
    else:
        T = -131.2+0.003*x
        P = 2.488*((T+273.1)/216.6)**(-11.4)

    rho = P/(0.29*(T+273.1)) # Density, ambient air (kg/m^3)
    Pa  = P*1000             # Pressure, ambient air (Pa)
    Ta  = T+273.1            # Temperature, ambient air (K)
    return Pa, rho, Ta

x  = np.arange(0, 1e5, 1) # Altitude (m)
Pa = atmo(x)[0]           # Atmospheric pressure (Pa)
Po = 50-Pa                # Pressure, lox, abs. (Pa)
Pf = 50-Pa                # Pressure, fuel, abs. (Pa)

# Required Mass

mp = 70           # Mass, propellant (kg)
mf = mp/(1+OF)    # Mass, fuel (kg)
mo = mp-mp/(1+OF) # Mass, lox (kg)

# Dome Volume

def dome(R):
    V = (4/3)*m.pi*R**3
    return V

# Cylinder volume

def cyl(R, L):
    V = m.pi*L*R**2
    return V

Endo = 2*dome(Ro) # End cap volume, lox (m^3)
Endf = 2*dome(Ro) # End cap volume, fuel (m^3)

CLo = rhoo*mo/cyl(Ro-t)-(8/3)*(Ro-t)    # Cyl. length, lox (m)
CLf = rhof*mf/cyl(Ro-t)-(8/3)*(Ro-t)    # Cyl. length, fuel (m)
Vo = mo/rhoo # Volume, lox (m^3)
Vf = mf/rhof # Volume, fuel (m^3)

mo = rhoo*Vo # Mass, lox (kg)
mf = rhof*Vf # mass, fuel (kg)
mp = mo+mf   # Mass, propellant (kg) 

# Stress, cylindrical portion

Sho = Po*Ro/t # Hoop stress, cyl, lox (Pa)
Shf = Pf*Ro/t # Hoop stress, cyl, fuel (Pa)
Sao = 0.5*Sho # Axial stress, cyl, lox (Pa)
Saf = 0.5*Shf # Axial stress, cyl, fuel (Pa)

# Stress, endcaps
# Note: Static equilibrium, stress net resultant

Sno = Sao # Stress, endcaps, lox (Pa)
Snf = Saf # Stress, endcaps, Fuel (Pa)

# Composite overwrap

# Parameters
# Note: typical El = 130e9 Pa
#               Et = 11e9 Pa
#               v  = 0.27

# Expected values, carbon composite (TEMP): 
#                         Vf  = 0.55
#                         E22 = 130 GPa
#                         E11 = 11 GPa
#                         G12 = 5.5 GPa
#                         v12 = 0.27

rhofib = 2500    # Density, fiber (kg/m^3) (TEMP)
rhom   = 1611    # Density, matrix (kg/m^3) (TEMP)
rhopla = 1000    # Density, internal plastic (kg/m^3) (TEMP)
wf     = 0.65    # Weight, fiber (kg) (TEMP)
wm     = 0.35    # Weight, matrix (kg) (TEMP)
Ef     = 106.3e9 # Modulus, fiber (Pa)
Em     = 3.4e9   # Modulus, matrix (Pa)

Vf = (wf/rhofib)/(wf/rhofib + wm/rhom) # Volume fraction, fiber
Vm = 1-Vf                              # Volume fraction, matrix

rhocom = rhofib*Vf + rhom*Vm # Density, composite (kg/m^3)

vf = 0.26 # Poisson's ratio, fiber (TEMP)
vm = 0.33 # Poisson's ratio, matrix (TEMP)

Gf = Ef/(2*(1+vf)) # Shear modulus, fiber (Pa)
Gm = Em/(2*(1+vm)) # Shear modulus, matrix (Pa)

# Weave Angle: Orientation of the specific referenced ply
# Wraps: - Cylindrical, one at 0 deg.
#        - Helical, one at 45 deg., one at -45 deg.

Th = 45 # Weave angle (deg.)

# Major Poisson's: Ratio of trans. contraction strain to long. extension strain in the direction of the stretch force
# Tensile --> positive

E22 = Ef*Em/(Em*Vf+Ef*Vm) # Modulus, longitudinal (Pa)
E11 = Ef*Vf+Em*Vm         # Modulus, transverse (Pa)
v12 = vf*Vf+vm*Vm         # Poisson's ratio, major
v21 = E22*v12/E11         # Poisson's ratio, minor
G12 = Gf*Gm/(Gf*Vm+Gm*Vf) # Shear modulus

# Windenburg and Trilling Equation
# Carbon Composite Epoxy Helix at 45 deg.

Ex  = E22/(m.cos(Th)**4 + (E22/E11)*m.sin(Th)**4 + 0.25*((E22/E11)-2*v12)*m.sin(2*Th)**2) # Modulus, x (Pa)
Ey  = E22/(m.sin(Th)**4 + (E22/E11)*m.cos(Th)**4 + 0.25*((E22/E11)-2*v12)*m.sin(2*Th)**2) # Modulus, y (Pa) 
Pcr = (2.42*Ex*(t/Do)**2.5)/(((1-v12**2)**0.75)*(L/Do - 0.45*m.sqrt(t/Do)))               # Critical pressure (Pa)

# Results

print('\n')
print('FUEL TANK PARAMETERS')
print('--------------------')
print('Tank Radius         = {0:.2f} m'.format(Ro))
print('Tank Wall Thickness = {0:.2f} mm'.format(t*1000))
print('Volume, lox         = {0:.3f} m^3'.format(Vo))
print('Volume, fuel        = {0:.3f} m^3'.format(Vf))
print('\nFUEL TANK MASS')
print('--------------')
print('Mass, Lox   = {0:.2f} kg'.format(mo))
print('Mass, Eth.  = {0:.2f} kg'.format(mf))
print('Mass, Prop. = {0:.2f} kg'.format(mp))

# Plotting
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
f, (ax1, ax2) = plot.subplots(2, sharex=True)
ax1.plot(x, Po)
ax1.yaxis.major.locator.set_params(nbins=6)
ax1.set_title('LV4, Lox Pressure (kPa) vs. Altitude (km)')
ax2.plot(x, Pf)
ax2.yaxis.major.locator.set_params(nbins=6)
ax2.set_title('LV4, Fuel Pressure (kPa) vs. Altitude (km)')
plot.show()



