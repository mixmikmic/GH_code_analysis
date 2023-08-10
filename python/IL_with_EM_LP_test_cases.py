import numpy as np
from IL_LP_with_EM import IL_LP_with_EM

import matplotlib.pyplot as plt

dt = 10
pr = np.array([10,20,30,40,50,60])
P = 20
E = 900
T_off = 4
small_const = 0.0001

# LP solution
solution = IL_LP_with_EM(dt, pr, P, E, T_off, small_const)
print('Total Cost: ', solution[0])
print('Power statuses: ', solution[1][:])
print('On/Off status: ', solution[2][:])

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([10,50,10,50,50])
P = 20
E = 500
T_off = 2
small_const = 0.0001

# LP solution
solution = IL_LP_with_EM(dt, pr, P, E, T_off, small_const)
print('Total Cost: ', solution[0])
print('Power statuses: ', solution[1][:])
print('On/Off status: ', solution[2][:])

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([10,50,2,10,50,1])
P = 13
E = 90
T_off = 2
small_const = 0.0001

# LP solution
solution = IL_LP_with_EM(dt, pr, P, E, T_off, small_const)
print('Total Cost: ', solution[0])
print('Power statuses: ', solution[1][:])
print('On/Off status: ', solution[2][:])

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([1,50,1,10,10])
P = 20
E = 500
T_off = 2
small_const = 0.0001

# LP solution
solution = IL_LP_with_EM(dt, pr, P, E, T_off, small_const)
print('Total Cost: ', solution[0])
print('Power statuses: ', solution[1][:])
print('On/Off status: ', solution[2][:])

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[1][:], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

