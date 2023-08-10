

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# file from: http://ds.iris.edu/ds/products/emc-prem/
# based on Dziewonski & Anderson (1981) http://adsabs.harvard.edu/abs/1981PEPI...25..297D
file = 'PREM_1s.csv'
radius, density = np.loadtxt(file, delimiter=',', usecols=(0,2), comments='#', unpack=True)

# Now let's plot these data
plt.plot(radius, density)
plt.xlabel('Radius (km)')
plt.ylabel('Density (g/cm$^3$)')

# create an array of needed zeros
x = np.zeros(len(radius))

truedensity = (density)*1000
trueradius = (radius)*1000

# looping over these, we save new ...
for i in range(1,len(radius)):
    x[i] = ((4/3)*np.pi*((trueradius[i-1]**3)-(trueradius[i])**3)*(truedensity[i])) + x[i-1]
    
totalmass=x[len(radius)-1]
print(totalmass)

plt.plot(x)

halfmass=totalmass*.5
R=np.where(np.abs(x-halfmass)==np.min(np.abs(x-halfmass)))
print(radius[R])
answer = radius[R]
wantedradius = radius[0] - answer
print(wantedradius)
wantedradius/radius[0]





