''' 
Let's import the Python modules we'll need for this example!
This normally goes at the top of your script or notebook, so that all code below it can see these packages.

By the way, the triple-quotes give you blocks of comments
'''
# while the pound (or hash) symbol just comments things

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# Let's read some data in

# file from: http://ds.iris.edu/ds/products/emc-prem/
# based on Dziewonski & Anderson (1981) http://adsabs.harvard.edu/abs/1981PEPI...25..297D
file = 'PREM_1s.csv'
radius, density = np.loadtxt(file, delimiter=',', usecols=(0,2), comments='#', unpack=True)

radius

density

# Now let's plot these data
plt.plot(radius, density)
plt.xlabel('Radius (km)')
plt.ylabel('Density (g/cm$^3$)')

siz=np.size(density)
n=np.zeros(np.size(density+1))

for i in range(0,siz-1):
    n[i]=4/3*np.pi*((radius[i])**3-(radius[i+1])**3)*density[i]*1000*10**9
x=np.arange(0,198,1)
plt.plot(x,n[x])

k=np.zeros(siz)
for i in range(0,siz):
    k[i]=np.sum(n[0:i])

plt.plot(radius[::-1],k)
plt.xlabel('Radius [km]')
plt.ylabel('Mass[kg]')

for i in range(0,199):
    if k[i]<.5*k[198]:
        flag=False
    if k[i]>.5*k[198]:
        print radius[i]
        break



