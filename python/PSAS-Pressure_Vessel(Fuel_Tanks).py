import math as m
from sympy import solveset

# Aluminum Pressure Vessel Dimensions
Pi = 10        # Internal pressure (Pa)
Po = 1         # External pressure (Pa)
ri = 1         # Inner radius (m)
ro = 1.1       # Outer radius (m)
r  = (ro-ri)/2 # Average radius (m)
t  = 0.1       # Wall thickness (m)
l  = 5         # Length (m)

# Aluminum Properties (1060)
rho_Al = 2700      # Material density (kg/m^3)
S_Al   = 255*10**6 # Yield stress - Al. (Pa)

# Misc. Parameters
FS = 1.1 # Factor of safety
g = 9.81 # Gravity (m/s^2)


# note** in python statements are always delimited by an ':'
if (r/t) <= 10: # Thin-Wall Condition 
    Hoop_Stress         = (Pi*r)/t
    Radial_Stress       = -Pi/2
    Longitudinal_Stress = (Pi*r)/(2*t)
else:
    Hoop_Stress         = (Pi*ri**2 - Po*ro**2 - ri**2*ro**2*(po-pi)/r**2)/(ro**2 - ri**2)
    Radial_Stress       = (Pi*ri**2 - Po*ro**2 + ri**2*ro**2*(po-pi)/r**2)/(ro**2 - ri**2)    
    Longitudinal_Stress = (Pi*ri**2)/(ro**2 - ri**2)

# where is sigma_name, what does it do?
Sigma = abs([Hoop_Stress, Radial_Stress, Longitudinal_Stress])

#solveset from the sympy package is like syms in matlab, but what are you solving for? hoop, radial and longitudnal stress already have values?!
[Hoop_Stress, Radial_Stress, Longitudinal_Stress, Sigma_Name] = solveset(Hoop_Stress, Radial_Stress, Longitudinal_Stress, Sigma_Name)
Max_Stress = max(Sigma)
Sigma_Max  = find(Sigma==Max_Stress) # Maximum stress
Max_Stress_Type = Sigma_Name(1, Sigma_Max)

if Sigma_Max<(S_Al*FS): # Stress Analysis
    print('Stress_Analysis:\n\n    Pass\n')
else:
    print('Stress_Analysis:\n\n Fail\n')

Volume_Al = pi*l*(ro**2-ri**2)+2*pi*ri**2 # Total Volume (m^3)
Mass_Al   = rho_Al*Volume_Al;             # Total Mass (kg)
Weight_Al = Mass_Al*g                     # Total Weight (N)

import math as m
# Carbon Fiber & Nomex Pressure Vessel
Pi     = 10   # Internal Pressure (Pa)
Po     = 1    # External Pressure (Pa)
t_CF   = 0.01 # Carbon Fiber Thickness (m)
t_Nom  = .01  # Nomex Thickness (m)
r_CF_i = 1    # Inner Radius (m)
r_CF_o = 1.1  # Outer Radius (m)
r_Nom  = 2    # Average Radius (m)
l      = 5    # Length (m)

rho_CF  = 2700      # Material Density (kg/m^3)
S_CF    = 300*10**6 # Yield Stress - Aluminum (Pa)
rho_Nom = 2700      # Material Density (kg/m^3)
S_Nom   = 150*10**6 # Yield Stress - Aluminum (Pa)

FS = 1.1  # Factor of Safety
g  = 9.81 # Gravity (m/s^2)

S_CF  = 30 # Yield Strength (N)
S_Nom = 13 # Initial Tear Strength (N)
'''
S_Total # Total Yield Stress (To be Determined)
Sigma_Total # Total Max Stress (To be Determined)
'''
Volume_CF = pi*l*t_CF*(r_CF_i**2+r_CF_o**2)# Volume of Carbon Fiber (m^3)
Volume_Nom = pi*l*t_Nom*r_Nom**2 # Volume of Nomex (m^3)
Mass_CF = rho_CF*Volume_CF # Mass of Carbon Fiber (kg)
Mass_Nom = rho_Nom*Volume_Nom # Mass of Nomex (kg)
Weight_Total =(Mass_CF+Mass_Nom)*g # Total Weight of Vessel (N)

