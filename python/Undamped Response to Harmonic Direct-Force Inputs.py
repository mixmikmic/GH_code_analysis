import numpy as np              # Grab all of the NumPy functions with nickname np

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
import matplotlib.pyplot as plt 

# Define the System Parameters
m = 1.0                 # kg
k = (2.0 * np.pi)**2.      # N/m (Selected to give an undamped natrual frequency of 1Hz)
wn = np.sqrt(k / m)       # Natural Frequency (rad/s)

# Set up input parameters
w = np.linspace(1e-6, wn*3, 1000)            # Frequency range for freq response plot, 0-3x wn with 1000 points in-between

x_amp = (1/m) / (wn**2 - w**2)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(x_amp) >= 5)
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
plt.ylabel(r'$ \frac{1}{m\left(\omega_n^2 - \omega^2\right)} $',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-1.0,1.0)
plt.xticks([1],['$\omega = \omega_n$'])
plt.yticks([0])


plt.plot(w/wn,x_amp,linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython Notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MassSpring_ForcedFreqResp_Amplitude.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

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
plt.ylabel(r'$\left| \frac{1}{m\left(\omega_n^2 - \omega^2\right)\right|} $',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(0.0,0.25)
plt.xticks([1],[r'$\omega = \omega_n'])
plt.yticks([0])


plt.plot(w/wn,x_mag, linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython Notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MassSpring_ForcedFreqResp_Magnitude.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Set up input parameters
wnorm = np.linspace(0,4,500)            # Frequency range for freq response plot, 0-4 Omega with 500 points in-between

x_amp = 1 / ((wn**2 * m) * (1 - wnorm**2))
xnorm_amp = x_amp * (m * wn**2)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(xnorm_amp) >= 100)
xnorm_amp[pos] = np.nan
wnorm[pos] = np.nan

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

plt.xlabel(r'Normalized Frequency $\left(\Omega\right)$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylabel(r'$\frac{m \omega_n^2}{\bar{f}} \bar{x}$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-4.0,4.0)
plt.xticks([0,1],['0','1'])
plt.yticks([0,1])

plt.plot(wnorm,xnorm_amp,linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MassSpring_ForcedFreqResp_NormAmp.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

xnorm_mag = np.abs(xnorm_amp)

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

plt.xlabel(r'Normalized Frequency $\left(\Omega\right)$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylabel(r'$\left| \frac{m \omega_n^2}{\bar{f}} \bar{x} \right|$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(0.0,5.0)
plt.xticks([0,1],['0','1'])
plt.yticks([0,1])

plt.plot(wnorm, xnorm_mag, linewidth=2)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# savefig('MassSpring_ForcedFreqResp_NormMag.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

