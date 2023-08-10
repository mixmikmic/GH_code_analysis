# Setup
from __future__ import print_function, absolute_import, division
import ionize
import pprint

from matplotlib import pyplot as plot
get_ipython().magic('matplotlib inline')

import numpy as np
np.set_printoptions(precision=3)

acid = ionize.Ion('myAcid', [-1], [5], [-25e-9])
base = ionize.Ion('myBase', [1], [8], [20e-9])
print(acid)        # The string includes only the class and name.
print(repr(base))  # The representation contains enough information to reconstruct the ion. 

print('myAcid Ka at (I=0 M) =', acid.acidity())
print('myAcid Ka at (I=0.5 M) =', acid.acidity(ionic_strength=0.5))

pH = np.linspace(0,14)

for I in [None, 0., 0.001, 0.01, 0.1]:
    mu = [base.mobility(p, I) for p in pH]
    if I is not None:
        label = 'I={} M'.format(I)
    else:
        label = 'I=None'
    plot.plot(pH, mu, label=label)

plot.xlabel('pH'); plot.xlim(0, 14)
plot.ylabel('effective mobility (m^2/v/s)'); plot.ylim(-.1e-8, 2.1e-8)
plot.legend()
plot.show()

db = ionize.Database()
histidine = db['histidine']
print(repr(histidine))

for ionic_strength in (None, 0):
    mu_histidine = [histidine.mobility(p, ionic_strength=ionic_strength) for p in pH]
    plot.plot(pH, mu_histidine, label="I={}".format(ionic_strength))
    
plot.xlabel('pH'); plot.xlim([0, 14])
plot.ylabel('effective mobility (m^2/v/s)')
plot.legend()
plot.show()

print("Search results for 'amino'\n--------------------------")
pprint.pprint(db.search('amino'))
print("\nSearch results for 'chloric'\n----------------------------")
pprint.pprint(db.search('chloric'))
print("\nSearch results for 'per'\n------------------------")
pprint.pprint(db.search('per'))
print('\nOh, copper is what I was looking for.')
print(db.load('copper'))

print(len(db.data), 'ions in database.')

buffer=ionize.Solution([db['tris'], db['chloride']], [0.1, 0.085])

print('pH =', buffer.pH)
print('I =', buffer.ionic_strength, 'M')
print('conductivity =', buffer.conductivity(), 'S/m')
print('buffering capacity =', buffer.buffering_capacity(), 'M')
print('debye length =', buffer.debye(), 'm')

sol = ionize.Solution(['bis-tris', 'acetic acid'], [0.1, 0.03])
print([ion.name for ion in sol.ions])
print(sol.concentration('acetic acid'))

c_tris = 0.1
c_hcl = np.linspace(0.0, 0.2, 50)
t_pH = [ionize.Solution(['tris', 'hydrochloric acid'], [c_tris, c_h]).pH for c_h in c_hcl]

plot.plot(c_hcl/c_tris, t_pH)
plot.xlabel('[HCl]/[Tris]')
plot.ylabel('pH')
plot.show()

water = ionize.Solution()
print('I =', water.ionic_strength, 'M')
print('pH =', water.pH)
print('conductivity =', water.conductivity(), 'S/m')

print('Stock:', buffer)
dilution = 0.5 * buffer + 0.5 * water
print('Dilution:', dilution)

buff = ionize.Solution(['tris'], 0.1)
print(buff.titrate('hydrochloric acid', 8.2))
print(buff.titrate('hydrochloric acid', 3))
print(buff.conductivity())
print(repr(buff.titrate('hydrochloric acid', 3, titration_property = 'conductivity')))
print(repr(buff.titrate('hydrochloric acid', 8)))



