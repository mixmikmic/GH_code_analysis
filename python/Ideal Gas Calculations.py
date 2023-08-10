n = 2.0
R = 8314.0
V = 22.4
T = 284.0

P = n*R*T/V

print P

get_ipython().system('pip install pint')

from pint import UnitRegistry
ur = UnitRegistry()

n = 2.0 * ur.mol
T = 284.0 * ur.kelvin
V = 22.4 * ur.liter
R = 8314 * ur.Pa *ur.liter/(ur.kelvin * ur.mol)

P = n*R*T/V

print P.to(ur.gallon)

print V.to(ur.gallon)



