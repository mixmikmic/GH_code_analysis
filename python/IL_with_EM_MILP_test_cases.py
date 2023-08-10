import numpy as np
from IL_MILP_with_EM import IL_MILP_with_EM

import matplotlib.pyplot as plt

dt = 10
pr = np.array([10,20,30,40,50,60])
P = 20
E = 900
T_off = 4
small_const = 0.0001
N = len(pr)

# MILP solution
solution = IL_MILP_with_EM(dt, pr, P, E, T_off, small_const)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0:N], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([10,50,10,50,50])
P = 20
E = 500
T_off = 2
small_const = 0.0001
N = len(pr)

# MILP solution
solution = IL_MILP_with_EM(dt, pr, P, E, T_off, small_const)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0:N], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([10,50,2,10,50,1])
P = 13
E = 90
T_off = 2
small_const = 0.0001
N = len(pr)

# MILP solution
solution = IL_MILP_with_EM(dt, pr, P, E, T_off, small_const)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0:N], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

dt = 10
pr = np.array([1,50,1,10,10])
P = 20
E = 500
T_off = 2
small_const = 0.0001
N = len(pr)

# MILP solution
solution = IL_MILP_with_EM(dt, pr, P, E, T_off, small_const)

# Plot the price distribution
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0:N], 'x-')
ax.plot()

ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

