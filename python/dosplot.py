#--------------------------------------------------------------------------------------------
# Import operator numpy and matplotlib
#--------------------------------------------------------------------------------------------
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

#--------------------------------------------------------------------------------------------
# Define a function myFloat transform myList to float
#--------------------------------------------------------------------------------------------
def myFloat(myList):
    return map(float, myList)

#--------------------------------------------------------------------------------------------
# Read input file DOSCAR
#--------------------------------------------------------------------------------------------
dos = [line.split() for line in open ('data/DOSCAR', 'r')]

nedos = int(dos[5][2])                  # Read NEDOS -- number of energy points
natom = int(dos[0][0])                  # Read number of atoms
ef = float(dos[5][3])                   # Read Fermi level

#--------------------------------------------------------------------------------------------
# Read the total and partial DOS
# Notice that VASP write the partial DOS in the sequence -
# s p_y p_z p_x d_xy d_yz d_z^2 d_xz d_x^2-y^2 (spin-up and down)
#--------------------------------------------------------------------------------------------
pdos = np.zeros((nedos, 19))
for i in range(natom):
    pdos = pdos + np.array(map(myFloat, dos[(nedos+1)*(i+1)+6:(nedos+1)*(i+2)+5]))
t_pdos = pdos.T
energy = t_pdos[0]/natom-ef             # Set fermi-level to zero

s_up = t_pdos[1]
p_up = t_pdos[3] + t_pdos[5] + t_pdos[7]
d_up = t_pdos[9] + t_pdos[11] + t_pdos[13] + t_pdos[15] + t_pdos[17]
t_up = s_up + p_up + d_up

s_d = -t_pdos[2]
p_d = -t_pdos[4] - t_pdos[6] - t_pdos[8]
d_d = -t_pdos[10] - t_pdos[12] - t_pdos[14] - t_pdos[16] - t_pdos[18]
t_d = s_d + p_d + d_d

#--------------------------------------------------------------------------------------------
# Plot the DOS
# Change the plot scale to the part you are interested
#--------------------------------------------------------------------------------------------

axes = plt.gca()
axes.set_xlim([-8,8])
axes.set_ylim([-30,30])                 # set x and y axis range 

plt.fill_between(energy, 0, t_up, color='lightgrey', alpha=0.5)
plt.fill_between(energy, 0, t_d, color='lightgrey', alpha=0.5)
plt.plot(energy, t_up, label='t', color='lightgrey', lw=2)
plt.plot(energy, t_d, color='lightgrey', lw=2)

plt.plot(energy, s_up, label='s', color='#a2142f', lw=2)
plt.plot(energy, p_up, label='p', color='#77ac30', lw=2)
plt.plot(energy, d_up, label='d', color='#0072bd', lw=2)

plt.plot(energy, s_d, color='#a2142f', lw=2)
plt.plot(energy, p_d, color='#77ac30', lw=2)
plt.plot(energy, d_d, color='#0072bd', lw=2)

plt.plot([0.0, 0.0], [-30.0, 30.0], 'k-', lw=1)
plt.plot([-10.0,10.0], [0.0, 0.0], 'k-', lw=1)
plt.legend(loc='upper left')
plt.show()

