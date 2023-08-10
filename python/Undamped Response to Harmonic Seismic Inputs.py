import numpy as np              # Grab all of the NumPy functions with nickname np

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
import matplotlib.pyplot as plt

# Define the System Parameters
m = 1.0                 # kg
k = (2.*np.pi)**2.      # N/m (Selected to give an undamped natural frequency of 1Hz)
wn = np.sqrt(k/m)       # Natural Frequency (rad/s)

# Set up input parameters
w = np.linspace(0,wn*3,500)            # Frequency range for freq response plot, 0-3x wn with 500 points in-between

x_amp = (wn**2) / (wn**2 - w**2)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(x_amp) >= 15)
x_amp[pos] = np.nan
w[pos] = np.nan

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.2,left=0.15,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel(r'Input Frequency $\left(\omega\right)$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylabel(r'$ \frac{\omega_n^2}{\omega_n^2 - \omega^2} $',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-10.0,10.0)
plt.yticks([0])
plt.xticks([0,1],['0','$\omega = \omega_n$'])

plt.plot(w/wn, x_amp, linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MassSpring_SeismicFreqResp_Amplitude.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Define the magnitude of the response
x_mag = np.abs(x_amp)

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.2,left=0.15,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel(r'Input Frequency $\left(\omega\right)$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylabel(r'$ \left|\frac{\omega_n^2}{\omega_n^2 - \omega^2} \right|$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(0.0,10.0)
plt.yticks([0])
plt.xticks([0,1],['0','$\omega = \omega_n$'])

plt.plot(w/wn,x_mag,linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# savefig('MassSpring_SeismicFreqResp_Magnitude.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Set up input parameters
w = np.linspace(0,wn*3,500)            # Frequency range for freq response plot, 0-3x wn with 500 points in-between

xddot_amp = -(wn**2 * w**2) / (wn**2 - w**2)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(xddot_amp) >= 600)
xddot_amp[pos] = np.nan
w[pos] = np.nan

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.20,left=0.15,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel(r'Input Frequency $\left(\omega\right)$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylabel(r'$ \frac{-\omega_n^2 \omega^2}{\omega_n^2 - \omega^2} $',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-500.0,500.0)
plt.yticks([0])
plt.xticks([0,1],['0','$\omega = \omega_n$'])

plt.plot(w/wn,xddot_amp,linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MassSpring_SeismicFreqResp_AccelAmplitude.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Set up input parameters
wnorm = np.linspace(0,3,500)            # Frequency range for freq response plot, 0-3x wn with 500 points in-between

xnorm_amp = (1) / (1 - wnorm**2)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(xnorm_amp) >= 15)
x_amp[pos] = np.nan
wnorm[pos] = np.nan

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.23,left=0.15,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel(r'Normalized Frequency $\left(\Omega = \frac{\omega}{\omega_n}\right)$',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$ \frac{1}{1 - \Omega^2} $',family='serif',fontsize=22,weight='bold',labelpad=15)
plt.ylim(-5.0,5.0)
plt.yticks([0,1])
plt.xticks([0,1],['0','$\Omega = 1$'])

plt.plot(wnorm, xnorm_amp, linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# savefig('MassSpring_SeismicFreqResp_NormAmp.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Take the absolute value to get the magnitude
xnorm_mag = np.abs(xnorm_amp)

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.23,left=0.15,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel(r'Normalized Frequency $\left(\Omega = \frac{\omega}{\omega_n}\right)$',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$ \left| \frac{1}{1 - \Omega^2} \right|$',family='serif',fontsize=22,weight='bold',labelpad=15)
plt.ylim(0.0,5.0)
plt.yticks([0,1])
plt.xticks([0,1],['0','$\Omega = 1$'])

plt.plot(wnorm,xnorm_mag,linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MassSpring_SeismicFreqResp_NormMag.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

