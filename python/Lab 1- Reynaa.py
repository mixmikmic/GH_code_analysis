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

# some beginning thoughts...
'''We will need to integrate over these shells hence the need to add them as we go. So to find what radial step encloses half 
Earth's mass we will need to just divide the sum by 2? Or will we need to develop an array for the sum? Then we need to locate
where the Earth's mass is located.'''

# create an array of 100 zeros
x = np.zeros(100)

# loop over these, and save something in each one
for i in range(1, 100):
    x[i] = i*2 + 5 + x[i-1]

p = density
equation=np.zeros(199)
#Rescale the density and radius array
rads = np.zeros(199)
for j in range(radius.size):
    rads[j] = radius[j]*100000.0
#The upper limit for the for loop should be 
for i in range(1,density.size):
    equation[i] = (4./3.)*np.pi*(rads[i-1]**3.-rads[i]**3.)*p[i]+equation[i-1]
    mass = np.sum(equation[i])

plt.plot(equation)

print mass
#According to Google, the mass of the Earth is 5.97*10^27 g which is awful close to the value found!

half_mass=(mass/2.0)
print half_mass

#Finding the radial step. Take whole array, subtract from half the mass and then using the minimal value to find the desired value.
total = np.abs(radius - half_mass) 

total_1 = radius[80]
print total_1


#U = radius[100]
#print U
#error = 6371 - U
#print error

plt.plot(x)

z = np.where((equation > half_mass))

plt.plot(radius, equation)
plt.scatter(radius[z], equation[z], c='red', lw=0)

#This corresponds to enclosed half mass which makes sense considering that there is this unique distribution of density.
radius[z][0]

