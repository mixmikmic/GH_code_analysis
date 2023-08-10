import numpy as np
from NL_LP import NL_LP

import matplotlib.pyplot as plt

dt = 10
pr = np.array([50, 10, 50, 10, 50])
L = 2
P = 20

# LP solution 
solution = NL_LP(dt, pr, L, P)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([50, 10, 50, 10, 50])
L = 4
P = 20

# LP solution 
solution = NL_LP(dt, pr, L, P)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([10, 20, 30, 40, 50, 60, 70])
L = 3
P = 71

# LP solution 
solution = NL_LP(dt, pr, L, P)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([10,50,2,10,50,1])
L = 3
P = 20

# LP solution 
solution = NL_LP(dt, pr, L, P)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([1,50,1,10,10])
L = 3
P = 20

# LP solution 
solution = NL_LP(dt, pr, L, P)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

