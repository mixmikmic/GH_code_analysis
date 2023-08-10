# enabling Jupyter mode for Matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# enabling seaborn interface for better aesthetics
plt.style.use('seaborn-whitegrid')

import numpy as np

fig = plt.figure()
ax = plt.axes()

#Plot data
fig = plt.figure()
x = [1,2,3,4,5,6,7,8,9,10]
plt.plot(x,[i**2 for i in x], label = "Data")

#Add labels
plt.xlabel("Days")
plt.ylabel("Days Squared")
plt.legend()

plt.show()

x = np.linspace(0, 10, 100)

print(x)

plt.plot(x, x + 1) # plotting f(x) = x + 1

plt.plot(x, np.sin(x)) # plotting f(x) = sin(x), the sine function is imported from NumPy

plt.plot(x, x + 2) # plotting f(x) = x + 2
plt.plot(x, np.cos(x)) # plotting f(x) = cos(x)

x = np.array([0, 3, 6, 4, 1])
y = np.array([10, 20, 16, 5, 8])

plt.scatter(x, y)

x = np.array([0, 3, 6, 4, 1])
y = np.array([10, 20, 16, 5, 8])

plt.scatter(x, y, c='red', s=100)

x = np.random.randn(100) # generate a random dataset of 1000 elements

x

plt.hist(x)

