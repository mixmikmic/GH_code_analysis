get_ipython().magic('pylab inline')
import pysd

model = pysd.read_vensim('../../models/Epidemic/SIR.mdl')

res = model.run(params={'Infectivity': 0.02})

res['Infected'].plot()
plt.xlabel('Days')
plt.ylabel('Number of Individuals Infected');

peak_value = res['Infected'].max()

print 'Up to', int(peak_value), 'individuals are infected at one time.'

import numpy as np
infectivity_values = np.arange(.005, .105, .005)
print infectivity_values

peak_value_list = []

for inf in infectivity_values:
    res = model.run(params={'Infectivity': inf})
    peak_value_list.append(res['Infected'].max())

peak_value_list

plt.plot(infectivity_values, peak_value_list)
plt.grid()
plt.xlabel('Infectivity')
plt.ylabel('Peak Value of Infections')
plt.title('Peak level of infection as a function of infectivity.');

