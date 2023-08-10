get_ipython().magic('matplotlib inline')
from numpy import *

mCO2_in = 34.5e9            # inflow, metric tonnes per year
mCO2_in = mCO2_in*1000      # inflow, kg per year
print("Global CO2 emissions = {:8.3g} kg/yr".format(mCO2_in))

nCO2_accum = 2.4e-6                   # accumulation, kg-mol CO2/kg-mol air/yr

mwAir = 28.97                         # kg air/kg-mol air
mwCO2 = 44.01                         # kg CO2/kg-mol CO2

wCO2_accum = nCO2_accum*mwCO2/mwAir   # kg CO2/kg air/yr

print("Accumulation Rate of CO2 = {:8.3g} kg CO2/kg air".format(wCO2_accum))

# Earth Radius in meters
R = 6371000       # m

# Earth Area in square meters
A = 4*pi*R**2     # m**2

# Mass of the atmosphere in kg
g = 9.81          # N/kg
P = 101325        # N/m**2
mAir = A*P/g      # kg

print("Estimated mass of the atmosphere = {:8.3g} kg".format(mAir))

mCO2_accum = wCO2_accum*mAir     # kg CO2/year

print("Change in CO2 = {:8.3g} kg CO2/year".format(mCO2_accum))

mCO2_out = mCO2_in - mCO2_accum

print("Global CO2 outflow = {:8.3g} kg CO2/yr".format(mCO2_out))

fCO2 = mCO2_accum/mCO2_in
print("Fraction of CO2 retained in the atmosphere = {:<.2g} ".format(fCO2))



