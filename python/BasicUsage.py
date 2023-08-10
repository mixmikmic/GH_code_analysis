get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
import numpy as np
import operator
import os, shutil
import pprint
from matplotlib import pyplot as plt
from IPython.display import Math
import pypropep as ppp

plt.style.use(u'ggplot')
plt.rcParams['figure.figsize'] = (10,6)

help(ppp.init)
ppp.init()

print ppp.PROPELLANT_FILE
print ppp.THERMO_FILE

# First copy the file
shutil.copy(ppp.PROPELLANT_FILE, 'propellant.dat')

# Now add a line
with open('propellant.dat', 'a') as f:
    f.write('    1099 AMMONIUM DINITRAMIDE            4H   4O  4N    0    0    0   -285 .0000]')

# Now re-init pypropep and note that there is one more propellant loaded than before 1031 vs 1030
ppp.init(propellant_file='propellant.dat')

# Get rid of our temporary propellant.dat file (if you want you can keep it around, this is just an example)
os.remove('propellant.dat')

# Now show that ADN is in the database:
pprint.pprint(ppp.find_propellant('ammonium di'))

Math(ppp.PROPELLANTS['AMMONIUM ALUMINUM PERCHLORATE'].formula(tex=True))

e = ppp.Equilibrium()
air = ppp.PROPELLANTS['NITROGEN (GASEOUS)']
e.add_propellant(air, 1.)
T_v = np.arange(100, 10000, 100)
U_v = np.zeros_like(T_v)
a_v = np.zeros_like(T_v)
for i in xrange(len(T_v)):
    e.set_state(P=1., T=T_v[i], type='TP')
    U_v[i] = e.properties.U
    a_v[i] = e.properties.Vson

ax = plt.subplot(211)
plt.plot(T_v, U_v)
plt.ylabel('U (kJ/kg)')

plt.subplot(212, sharex=ax)
plt.plot(T_v, a_v)
plt.xlabel('T (K)')
plt.ylabel('a (m/s)')

e.set_state(P=1., T=273, type='TP')
print e
print 'Composition: ',
pprint.pprint(e.composition)

e2 = ppp.Equilibrium()
h2o = ppp.PROPELLANTS['WATER']
co = ppp.PROPELLANTS['CARBON MONOXIDE']
e2.add_propellants([(h2o, 1.), (co, 1.)])
print e2

e2.set_state(P=1., T=500., type='TP')
print e2
pprint.pprint(e2.composition_sorted[0:8])

e2.reset()
Ru = 8.314     #kJ / kmol-K
T = np.linspace(500, 1000, 50)
comp = {
    "H2O" : [],
    "CO2" : [],
    "CO"  : [],
    "H2"  : []
}
# eq_const = []
# e2.set_state(P=1., T=300, type='TP')
# G0 = e2.properties.G
for i in xrange(len(T)):
    e2.set_state(P=1., T=T[i], type='TP')
    comp["H2O"].append(e2.composition["H2O"])
    comp["CO2"].append(e2.composition["CO2"])
    comp["CO"].append(e2.composition["CO"])
    comp["H2"].append(e2.composition["H2"])
#     eq_const.append(np.exp(-(G0-e2.properties.G) * (e2.properties.M) / Ru / T[i]))

# ax = plt.subplot(211)
for s,c in comp.items():
    plt.plot(T, c, label=s)
plt.ylabel('Composition (mol)')
plt.legend(loc='best')

# plt.subplot(212,sharex=ax)
# plt.plot(T, eq_const)
# plt.xlabel('Temperature (K)')
# plt.ylabel('Equilibrium constant')

ch4 = ppp.PROPELLANTS['METHANE']
air = ppp.PROPELLANTS['AIR (DRY AT SEA LEVEL)']
species = ['H2', 'H2O', 'O2', 'CO2', 'CO']
species_log = ['NO', 'H', 'O', 'OH']
phi = np.linspace(0.5, 1.5, 50)
T_adiabatic = []
N_O_air = air.atoms_of('O')
N_C_ch4 = ch4.atoms_of('C')
N_H_ch4 = ch4.atoms_of('H')
N_ratio_st = N_O_air / (N_C_ch4 * 2 + N_H_ch4 / 2)

comp = dict()
for s in (species + species_log):
    comp[s] = []
for i in xrange(len(phi)):
    e3 = ppp.Equilibrium()
    n_fuel = phi[i] * N_ratio_st
    e3.add_propellants([(ch4, n_fuel), (air, 1.)])
    e3.set_state(P=1., type='HP')
    T_adiabatic.append(e3.properties.T)
    for s in (species + species_log):
        comp[s].append(e3.composition[s])

plt.figure(figsize=(10,12))
ax = plt.subplot(311)
plt.plot(phi, T_adiabatic)
plt.ylabel("Temperature (K)")

plt.subplot(312,sharex=ax)
for s in species:
    plt.plot(phi, comp[s], label=s)
plt.ylabel("Composition")
plt.legend(loc='best', fontsize='small')

plt.subplot(313,sharex=ax)
for s in (species + species_log):
    plt.semilogy(phi, comp[s], label=s)
plt.xlabel(r"$\phi$")
plt.ylabel("Composition")
plt.legend(loc='best', fontsize='small')



