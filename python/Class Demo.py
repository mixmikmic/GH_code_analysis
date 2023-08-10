# Comment

V = 22.4
T = 284.
R = 8314.
n = 2.

P = n*R*T/V

print "Pressure = ", P, "pascals"

get_ipython().system('pip install pint')

from pint import UnitRegistry
ur = UnitRegistry()

V = 22.4 * ur.liter
T = 284. * ur.kelvin
n = 2.0 * ur.mol
R = 8314.0 * ur.Pa * ur.liter/ur.kelvin/ur.mol

P = n*R*T/V
print P

print P.to(ur.psi)

print P.to(ur.atm)



