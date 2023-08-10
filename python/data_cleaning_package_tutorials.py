import import_data
import sort_data

import_data.single_pd_matlab_data('converted_PL0.mat')

charge, discharge = sort_data.charge_discharge('converted_PL05.mat')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(1, 2, figsize = (14, 6), dpi = 600)
#plt.figure(figsize = (6, 6))
for i in range(24, 899, 36):
    if i in charge.keys():
        ax[0].plot((charge[i]['time']-charge[i]['time'].iloc[0]), charge[i]['voltage'])
    else:
        pass

for i in range(24, 899, 36):
    if i in discharge.keys():
        ax[1].plot((discharge[i]['time']-discharge[i]['time'].iloc[0]), discharge[i]['voltage'])
    else:
        pass


ax[0].set_xlabel('Time (seconds)', fontsize = 14)
ax[0].set_ylabel('Voltage (V)', fontsize = 14)
ax[0].set_title('Partial Charge Curves between 40 and 60% SoC', fontsize = 15)
ax[0].tick_params(axis='both', which='major', labelsize=12)

ax[1].set_xlabel('Time (seconds)', fontsize = 14)
ax[1].set_ylabel('Voltage (V)', fontsize = 14)
ax[1].set_title('Partial Disharge Curves between 40 and 60% SoC', fontsize = 15)
ax[1].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

plt.show()
plt.savefig("Discharge_&Charge.png")



