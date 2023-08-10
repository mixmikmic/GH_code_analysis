Q_btu = 50e6    # BTU

price = 0.08               # $ per kwh
btu_per_joule = 9.486e-4   # conversion factor
kwh_per_joule = 2.778e-7   # conversion factor

cost = price*kwh_per_joule*Q_btu/btu_per_joule

print("Winter heating cost = $", round(cost,2), "USD")

V_ft3 = Q_btu/1000    # ft^3

R = 10.73          # ft^3 psia/(lbmol R)
T_degC = 15        # deg C

# convert temperature to absolute
T_degR = 9.0*T_degC/5.0 + 491.67   # deg R

# compute lb moles
n_lbmol = 14.696*V_ft3/(R*T_degR)

print(round(T_degR,1), "degrees Rankine")
print(round(n_lbmol,1), "lb-mols")

V_ft3 = n_lbmol*R*T_degR/1000.0

m3_per_ft3 = 0.028317

V_m3 = V_ft3*m3_per_ft3

print("Storage volume of natural gas at 1000 psia and 15 deg C =", round(V_m3,1), "cubic meters")

btu_per_joule = 9.486e-4

Q_joule = Q_btu/btu_per_joule
print("Heat requirement =", round(Q_joule/1e6,1), "Megajoules")

m_kg = Q_joule/46.3e6
print("Mass of propane required = {0:.2f} kg".format(m_kg))

V_m3 = m_kg/0.493/1000.0
print("Volume of propane required = {0:.2f} cubic meters".format(V_m3))



