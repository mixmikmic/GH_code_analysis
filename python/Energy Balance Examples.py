Tin = 45                   # deg F
Tout = 105                 # deg F
deltaT = (Tout-Tin)/1.8    # convert difference to deg C
print "Tout - Tin = ", deltaT, "deg C"

Cp = 75.4                  # J/gmol/K from Murphy textbook
MW = 18.01                 # molecular weight
Cp = Cp/MW                 # convert to J/g/K or kJ/kg/C
print "Heat Capacity = ", Cp, "kJ/kgC"

F = 2.5                    # flow in gallons per minute
F = F*3.78541/60.0         # convert flow to liters per sec
rho = 1.0                  # density in kg/liter
mdot = rho*F               # mass flow in kg/sec
print "Mass flow = ", mdot, "kg/sec"

Qe = mdot*Cp*deltaT/0.9
print "Electrical Power = ", Qe, "kW"

from iapws import IAPWS97

# State 1: Starting Condition (x = vapor fraction)

V1_liq = 0.2*100.0/1000.0        # cubic meters
V2_vap = 0.8*100.0/1000.0        # cubic meters

m1_liq = V1_liq/IAPWS97(P=0.1,x=0.0).v
m1_vap = V2_vap/IAPWS97(P=0.1,x=1.0).v

print "liq. mass = ", m1_liq, "kg"
print "vap. mass = ", m1_vap, "kg"

m = m1_liq + m1_vap

print "    Total = ", m, "kg"

u1 = m1_liq*IAPWS97(P=0.1,x=0.0).u +  m1_vap*IAPWS97(P=0.1,x=1.0).u
print "U, sat. liq., at 1 bar = ", IAPWS97(P=0.1,x=0.0).u, "kJ/kg"
print "U, sat. vap., at 1 bar = ", IAPWS97(P=0.1,x=1.0).u, "kJ/kg"
print "Total internal energy = ", u1, "kJ"

print "Volume of ", m, "kg at sat. liq at 10 bar = ", m*100.0*IAPWS97(P=1.0,x=0.0).v, "liters"
print "Volume of ", m, "kg at sat. liq at 10 bar = ", m*100.0*IAPWS97(P=1.0,x=1.0).v, "liters"

m2_liq = (m*IAPWS97(P=1.0,x=1.0).v - 0.1)/(IAPWS97(P=1.0,x=1.0).v - IAPWS97(P=1.0,x=0.0).v)
m2_vap = m - m2_liq

print "liq. mass = ", m2_liq, "kg"
print "vap. mass = ", m2_vap, "kg"

print "    Total = ", m, "kg"

u2 = m2_liq*IAPWS97(P=1.0,x=0.0).u +  m2_vap*IAPWS97(P=1.0,x=1.0).u
print "U, sat. liq., at 1 bar = ", IAPWS97(P=1.0,x=0.0).u, "kJ/kg"
print "U, sat. vap., at 1 bar = ", IAPWS97(P=1.0,x=1.0).u, "kJ/kg"
print "Total internal energy = ", u2, "kJ"

print "Temperature at 10 bar = ", IAPWS97(P=1.0,x=1.0).T - 273.15, "deg C"
print "Energy required Q = U2 - U1 = ", u2-u1, "kJ"
print "Liq. Volume Fraction = ", m2_liq*IAPWS97(P=1.0,x=0.0).v/0.1
print "Vap. Volume Fraction = ", m2_vap*IAPWS97(P=1.0,x=1.0).v/0.1



