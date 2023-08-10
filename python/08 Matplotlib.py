get_ipython().magic('matplotlib inline')

# Line Plot
import numpy as np
import matplotlib.pyplot as plt
myArray = np.array([4,5,6])
plt.plot(myArray)
plt.xlabel('Some x-axis label')
plt.ylabel('Some y-axis label')
plt.title('Line Plot')
plt.show()

# A bar chart
y = np.random.rand(5)
x = np.arange(5)
plt.xlabel('Item')
plt.ylabel('Value')
plt.bar(x,y)
plt.title('Bar chart')
plt.show()

# A bar chart multiple inputs

N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, color='r')
p2 = plt.bar(ind + width, womenMeans, width, color='y')

plt.ylabel('Scores')
plt.title('A bar chart stacked')
plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()

# A bar chart stacked

N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, color='r')
p2 = plt.bar(ind, womenMeans, width, color='y', bottom=menMeans)

plt.ylabel('Scores')
plt.title('A bar chart stacked')
plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()

# Histogram
from numpy.random import normal
gaussian_numbers = normal(size=1000)
plt.hist(gaussian_numbers, bins=(-10,-1,1,10)) # Set bin bounds
plt.show()

# Scatter Plot
N = 50
x = np.random.rand(N)
y = np.random.rand(N)

plt.scatter(x, y)
plt.xlabel('Some x-axis label')
plt.ylabel('Some y-axis label')
plt.title('Scatter Plot')
plt.show()

# A scatter plot with colours, area and alpha blending
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

plt.scatter(x, y, s=area, c=colors, alpha=0.5)

plt.colorbar()

plt.show()



