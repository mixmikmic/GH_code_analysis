import numpy as np
from TCL_MILP_Testing import TCL_MILP
from All_without_Energy_Limit_MILP import ALL_without_Energy_Limit_MILP

import matplotlib.pyplot as plt

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

# Parameter Setting
pr = np.array([10, 50, 10, 50, 10])
dt = 10
# For NL
L = 3
P_NL = 20
# For IL
P_IL = 20
E_IL = 500
T_off = 3
Pmin = 0.4 * P_IL
# For TCL
P_TCL = 10
c_water = 0.073
m = 50
temp_up = 25
temp_o = 20
temp_req = 60
temp_en = np.array([25, 20, 18, 15, 15])
di = np.array([25, 27, 28, 22, 25])

# MILP Solution
solution = ALL_without_Energy_Limit_MILP(dt, pr, L, P_NL, P_IL, E_IL, T_off, Pmin, P_TCL, c_water, m, temp_up, temp_o, temp_req, temp_en, di)

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

