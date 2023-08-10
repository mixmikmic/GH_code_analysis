import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# User configurable
freq = 1/60
end_time = 180
v_peak = 220
step_size = 0.01
# Find the three-phase voltages
v1 = []
v2 = []
v3 = []
thetas = 2 * np.pi * freq * np.arange(0,end_time,step_size)
for ii, t in enumerate(thetas):
    v1.append(v_peak * np.sin(t))
    v2.append(v_peak * np.sin(t - (2/3)*np.pi))
    v3.append(v_peak * np.sin(t - (4/3)*np.pi))
# Plot the results
plt.plot(v1, label="V1")
plt.plot(v2, label="V2")
plt.plot(v3, label="V3")
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend(ncol=3);

# Theta is phase angle, each data should be a numpy array
def dq0_transform(thetas, data1, data2, data3):
    d_list = []
    q_list = []
    for ii, t in enumerate(thetas):
        # Clarke Transform
        x = (2*data1[ii] - data2[ii] - data3[ii])*(1/np.sqrt(6));
        y = (data2[ii] - data3[ii])*(1/np.sqrt(2));
        z = (data1[ii] + data2[ii] + data3[ii])*(1/np.sqrt(3));
        # Park Transform
        cos_t = np.cos(t);
        sin_t = np.sin(t);
        # CP Transform
        d = cos_t*x + sin_t*y;
        q = cos_t*y - sin_t*x;  
        # Add the results to the list
        d_list.append(d)
        q_list.append(q)
    return d_list, q_list

# Calculate and plot the results
d_list, q_list = dq0_transform(thetas, v1, v2, v3)
plt.plot(d_list)
plt.plot(q_list)
plt.xlabel('D Phase')
plt.ylabel('Q Phase');

