import math as m
import numpy as np

# Cradle Analysis, Quick
#  Geometry
Tdo = 0.5 # Tube, Outer Diameter [in.]
Tdi = 0.38 # Tube, Inner Diameter [in.]
Pw = 2 # Plate Width [in.]
Pt = 0.1 # Plate Thickness [in.]
Pdi = 5.4 # Plate, Inner Y Dist. from X=0 [in.]
Pdo = Pdi+Pt # Plate, Outer Y Dist. from X=0 [in.]
Tdc = 0.5*(4.58+0.25) # Tube, center to Y=0 Dist. [in.]

#  Material
E = 400e3 # Elastic Modulus [psi]

# Moment of Inertia (MoI)
#  Tube:
Tube_Ixo = (m.pi/4)*(Tdo/2)**4 # Outer Tube MoI (Relative to Tube Center)
Tube_Ixi = (m.pi/4)*(Tdi/2)**4 # Inner Tube MoI (Relative to Tube Center)
Tube_Ix = Tube_Ixo-Tube_Ixi # Tubing MoI [in.^4]
#  Plate:
Plate_Ix = Pw*Pt*(Pdi+0.5*Pt)**2 + (Pw*Pt**3)/12 # Plate MoI, x [in.^4]
Plate_Iy = (Pt*Pw**3)/12 # Plate MoI, y [in.^4]
#  Parallel Axis Theorem:
Tube_Ix_Corrected = Tube_Ix+(Tdc**2)*(m.pi*(0.5*Tdo)**2 - m.pi*(0.5*Tdi)**2) # Tube MoI [in.^4] 
Total_Ix = 2*Tube_Ix_Corrected + 2*Plate_Ix # Total MoI [in.^4]

print('Moment of Inertia about x-axis')
print(' Plate: {0:.4f} [in.^4]'.format(Plate_Ix))
print(' Tube:  {0:.4f} [in.^4]'.format(Tube_Ix_Corrected))
print(' Total: {0:.3f} [in.^4]'.format(Total_Ix))

#  SolidWorksI = np.array[125.99, 17.36, 123.99] # Solidworks MoI, [Ix, Iy, Iz]

# Bolt Shear
#  Material Properties
E = 28000e3 # Elastic Modulus, Stainless [psi]
S = 12500e3 # Shear Modulus, Stainless [psi]

#  Geometry
Rbolt = 0.112/2 # Bolt Radius [in.]
BoltA = m.pi*Rbolt**2 # Bolt x-section Area [in.^2]
N = 12 # Quantity of Bolts

#  G-Forces
g = 9.81 # Gravitational Acceleration [m/s^2]
a = 15*g  # Acceleration Expected [m/s^2]
mass = 1.25  # Mass [kg]
FgNewton = mass*a # G-Force [N]
Fg = FgNewton*0.2248 # G-Force [lbf]
Shear = Fg/(N*BoltA) # Shear Stress, per bolt [psi]

print('')
print('Shear Stresses:')
print(' G-Forces:        {0:.3f} [N]'.format(Fg))
print(' Total Bolt Area: {0:.4f} [in.^2]'.format(BoltA))
print(' Per Bolt:        {0:.2f} [psi]'.format(Shear))



