get_ipython().magic('pylab inline')
import pysd
import pandas as pd

emissions = pd.read_csv('../../data/Climate/global_emissions.csv', 
                        skiprows=2, index_col='Year',
                        names=['Year', 'Total Emissions', 
                               'Gas Emissions', 'Liquid Emissions', 
                               'Solid Emissions', 'Cement Emissions', 
                               'Flare Emissions', 'Per Capita Emissions'])
emissions.head()

model = pysd.read_vensim('../../models/Climate/Atmospheric_Bathtub.mdl')

print 'initial:', model.components.initial_time() 
print 'final:', model.components.final_time()

print 'initial:', emissions.index[0]
print 'final:', emissions.index[-1] 

res = model.run(initial_condition=(emissions.index[0], 
                                   {'Excess Atmospheric Carbon': 0}),
                return_timestamps=emissions.index.values,
                return_columns=['Emissions', 'Excess Atmospheric Carbon'])
res.head()

res = model.run(initial_condition=(emissions.index[0], 
                                   {'Excess Atmospheric Carbon': 0}),
                return_timestamps=emissions.index.values,
                return_columns=['Emissions', 'Excess Atmospheric Carbon'],
                params={'Emissions': emissions['Total Emissions']})

res.plot();



