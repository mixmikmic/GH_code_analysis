import numpy as np
from TCL_MILP import TCL_MILP

import matplotlib.pyplot as plt

pr = np.array([10, 20, 30, 40, 50])
dt = 10
P = 20
c_water = 0.073
m = 50
temp_up = 25
temp_o = 25
temp_req = 37
temp_en = np.array([25, 20, 18, 15, 15])
di = np.array([25, 27, 28, 22, 25])

# MILP Solution
solution = TCL_MILP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di)

# Plot
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0][:], 'x-')
ax.plot(solution[2][:], 'v-')
ax.plot(solution[3][:], 'm-')
ax.plot(solution[1][:], 'p-')
ax.text(4.5,0,'power status',color='orange')
ax.text(4.5,10,'price',color='blue')
ax.text(4.5,20,'lower bound',color='red')
ax.text(4.5,30,'optimal energy',color='purple')
ax.text(4.5,40,'upper bound',color='green')
ax.plot()
ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

# Compared with Case 1, only tmp_o changes
pr = np.array([10, 20, 30, 40, 50])
dt = 10
P = 20
c_water = 0.073
m = 50
temp_up = 25
temp_o = 20
temp_req = 37
temp_en = np.array([25, 20, 18, 15, 15])
di = np.array([25, 27, 28, 22, 25])

# MILP Solution
solution = TCL_MILP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di)

# Plot
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0][:], 'x-')
ax.plot(solution[2][:], 'v-')
ax.plot(solution[3][:], 'm-')
ax.plot(solution[1][:], 'p-')
ax.text(4.5,0,'power status',color='orange')
ax.text(4.5,10,'price',color='blue')
ax.text(4.5,20,'lower bound',color='red')
ax.text(4.5,30,'optimal energy',color='purple')
ax.text(4.5,40,'upper bound',color='green')
ax.plot()
ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

# Compared with case 2, only the price changes
pr = np.array([50, 40, 30, 20, 10])
dt = 10
P = 20
c_water = 0.073
m = 50
temp_up = 25
temp_o = 20
temp_req = 37
temp_en = np.array([25, 20, 18, 15, 15])
di = np.array([25, 27, 28, 22, 25])

# MILP Solution
solution = TCL_MILP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di)

# Plot
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0][:], 'x-')
ax.plot(solution[2][:], 'v-')
ax.plot(solution[3][:], 'm-')
ax.plot(solution[1][:], 'p-')
ax.text(4.5,0,'power status',color='orange')
ax.text(4.5,10,'price',color='blue')
ax.text(4.5,20,'lower bound',color='red')
ax.text(4.5,30,'optimal energy',color='purple')
ax.text(4.5,40,'upper bound',color='green')
ax.plot()
ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

# Compared with case 2 & 3, only the price changes
pr = np.array([10, 50, 10, 50, 10])
dt = 10
P = 20
c_water = 0.073
m = 50
temp_up = 25
temp_o = 20
temp_req = 37
temp_en = np.array([25, 20, 18, 15, 15])
di = np.array([25, 27, 28, 22, 25])

# MILP Solution
solution = TCL_MILP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di)

# Plot
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0][:], 'x-')
ax.plot(solution[2][:], 'v-')
ax.plot(solution[3][:], 'm-')
ax.plot(solution[1][:], 'p-')
ax.text(4.5,0,'power status',color='orange')
ax.text(4.5,10,'price',color='blue')
ax.text(4.5,20,'lower bound',color='red')
ax.text(4.5,30,'optimal energy',color='purple')
ax.text(4.5,40,'upper bound',color='green')
ax.plot()
ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

# Compared with case 4, only the power limit differs
pr = np.array([10, 50, 10, 50, 10])
dt = 10
P = 5
c_water = 0.073
m = 50
temp_up = 25
temp_o = 20
temp_req = 37
temp_en = np.array([25, 20, 18, 15, 15])
di = np.array([25, 27, 28, 22, 25])

# MILP Solution
solution = TCL_MILP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di)

# Plot
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0][:], 'x-')
ax.plot(solution[2][:], 'v-')
ax.plot(solution[3][:], 'm-')
ax.plot(solution[1][:], 'p-')
ax.text(4.5,0,'power status',color='orange')
ax.text(4.5,10,'price',color='blue')
ax.text(4.5,20,'lower bound',color='red')
ax.text(4.5,30,'optimal energy',color='purple')
ax.text(4.5,40,'upper bound',color='green')
ax.plot()
ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

# Compared with case 4, only the required temperature differs
pr = np.array([10, 50, 10, 50, 10])
dt = 10
P = 10
c_water = 0.073
m = 50
temp_up = 25
temp_o = 20
temp_req = 60
temp_en = np.array([25, 20, 18, 15, 15])
di = np.array([25, 27, 28, 22, 25])


# MILP Solution
solution = TCL_MILP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di)

# Plot
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0][:], 'x-')
ax.plot(solution[2][:], 'v-')
ax.plot(solution[3][:], 'm-')
ax.plot(solution[1][:], 'p-')
ax.text(4.5,0,'power status',color='orange')
ax.text(4.5,20,'price',color='blue')
ax.text(4.5,40,'lower bound',color='red')
ax.text(4.5,60,'optimal energy',color='purple')
ax.text(4.5,80,'upper bound',color='green')
ax.plot()
ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

# Compared with case 4, only the price changes
pr = np.array([10, 50, 1, 70, 15])
dt = 10
P = 20
c_water = 0.073
m = 50
temp_up = 25
temp_o = 20
temp_req = 37
temp_en = np.array([25, 20, 18, 15, 15])
di = np.array([25, 27, 28, 22, 25])

# MILP Solution
solution = TCL_MILP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di)

# Plot
fig, ax = plt.subplots()
ax.plot(pr, 'o-')
ax.plot(solution[0][:], 'x-')
ax.plot(solution[2][:], 'v-')
ax.plot(solution[3][:], 'm-')
ax.plot(solution[1][:], 'p-')
ax.text(4.5,0,'power status',color='orange')
ax.text(4.5,10,'price',color='blue')
ax.text(4.5,20,'lower bound',color='red')
ax.text(4.5,30,'optimal energy',color='purple')
ax.text(4.5,40,'upper bound',color='green')
ax.plot()
ax.set(xlabel='Time', ylabel='Price, $ (dots)/ \n Consumption, Wh (crosses)',
       title='Prices and Load Scheduling');

