get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# Alligator data snout length and weight
alligatorLength = np.array([3.87, 3.61, 4.33, 3.43, 3.81, 3.83, 3.46, 3.76, 3.50, 3.58, 4.19, 3.78, 3.71, 3.73, 3.78])
alligatorWeight = np.array([4.87, 3.93, 6.46, 3.33, 4.38, 4.70, 3.50, 4.50, 3.58, 3.64, 5.90, 4.43, 4.38, 4.42, 4.25])

# Find the slope and intercept of the best fit line
slope, intercept = np.polyfit(alligatorLength, alligatorWeight, 1) # polyfit can also do Polynomial regress
print 'slope: %f intercept: %f'% (slope, intercept)

# Create abline_values array to be able to plot the best fit line
abline_values = [slope * i + intercept for i in alligatorLength]

# Scatter Plot
plt.scatter(alligatorLength, alligatorWeight, c='g') # c='g' for green
plt.plot(alligatorLength, abline_values, c='b') # c='b' for blue
plt.xlabel('Alligator Snout Length')
plt.ylabel('Alligator Weight')
plt.title('Alligator')
plt.show()

# Using the slope and intercept we can get the likely weight of an Alligator
print 'Alligator Snout Length: %f Calculated weight: %f'% (4.0, slope * 4.0 + intercept)
print 'Alligator Snout Length: %f Calculated weight: %f'% (4.2, slope * 4.2 + intercept)

