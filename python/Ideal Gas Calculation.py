get_ipython().system('pip install pint')

from pint import UnitRegistry
ur = UnitRegistry()

Rgas = 8.314 * ur.joule/ur.kelvin/ur.mole
T = 284 * ur.kelvin
n = 2 * ur.mole
V = 22.4 * ur.liter

P = n*Rgas*T/V
print P.to(ur.psi)



