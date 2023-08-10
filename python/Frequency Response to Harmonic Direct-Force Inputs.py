import numpy as np              # Grab all of the NumPy functions with nickname np

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
import matplotlib.pyplot as plt

# Define the System Parameters
m = 1.0                 # kg
k = (2.0 * np.pi)**2    # N/m (Selected to give an undamped natrual frequency of 1Hz)
wn = np.sqrt(k / m)     # Natural Frequency (rad/s)

z = 0.25                # Define a desired damping ratio
c = 2 * z * wn * m      # calculate the damping coeff. to create it (N/(m/s))

# Set up input parameters
wun = np.linspace(0,5,500)          # Frequency range for freq response plot, 0-4 Omega with 500 points in-between
w = np.linspace(0,5,500)            # Frequency range for freq response plot, 0-4 Omega with 500 points in-between

# Let's examine a few different damping ratios
z = 0.0
mag_normal_un = 1/(k*np.sqrt((1 - w**2)**2 + (2*z*w)**2))
phase_un = -np.arctan2((2*z*w),(1 - w**2)) * 180/np.pi

# Let's mask the phase discontinuity, so it isn't plotted
pos = np.where(np.abs(k*mag_normal_un) >= 25)
phase_un[pos] = np.nan
wun[pos] = np.nan


z = 0.1
mag_normal_0p1 = 1/(k*np.sqrt((1 - w**2)**2 + (2*z*w)**2))
phase_0p1 = -np.arctan2((2*z*w),(1 - w**2)) * 180/np.pi


z = 0.2
mag_normal_0p2 = 1/(k*np.sqrt((1 - w**2)**2 + (2*z*w)**2))
phase_0p2 = -np.arctan2((2*z*w),(1 - w**2)) * 180/np.pi


z = 0.4
mag_normal_0p4 = 1/(k*np.sqrt((1 - w**2)**2 + (2*z*w)**2))
phase_0p4 = -np.arctan2((2*z*w),(1 - w**2)) * 180/np.pi

# Let's plot the magnitude (normlized by k G(Omega))

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='Serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='Serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel(r'Normalized Frequency ($\Omega$)',family='Serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$k |G(\Omega)|$',family='Serif',fontsize=22,weight='bold',labelpad=35)

plt.plot(wun, k*mag_normal_un, linewidth=2, label=r'$\zeta$ = 0.0')
plt.plot(w, k*mag_normal_0p1, linewidth=2, linestyle = '-.', label=r'$\zeta$ = 0.1')
plt.plot(w, k*mag_normal_0p2, linewidth=2, linestyle = ':', label=r'$\zeta$ = 0.2')
plt.plot(w, k*mag_normal_0p4, linewidth=2, linestyle = '--',label=r'$\zeta$ = 0.4')

plt.xlim(0,5)
plt.ylim(0,7)

leg = plt.legend(loc='upper right', fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='Serif',fontsize=16)

# save the figure as a high-res pdf in the current folder
# plt.savefig('Forced_Freq_Resp_mag.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Now let's plot the phase


# Plot the Phase Response
# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='Serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='Serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel(r'Normalized Frequency ($\Omega$)',family='Serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Phase (deg.)',family='Serif',fontsize=22,weight='bold',labelpad=8)

plt.plot(wun, phase_un, linewidth=2, label=r'$\zeta$ = 0.0')
plt.plot(w, phase_0p1, linewidth=2, linestyle = '-.', label=r'$\zeta$ = 0.1')
plt.plot(w, phase_0p2, linewidth=2, linestyle = ':', label=r'$\zeta$ = 0.2')
plt.plot(w, phase_0p4, linewidth=2, linestyle = '--', label=r'$\zeta$ = 0.4')

plt.xlim(0,5)
plt.ylim(-190,10)
plt.yticks([-180,-90,0])

leg = plt.legend(loc='upper right', fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='Serif',fontsize=16)

# save the figure as a high-res pdf in the current folder
# plt.savefig('Forced_Freq_Resp_Phase.pdf',dpi=300)

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Let's plot the magnitude and phase as subplots, to make it easier to compare

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(8,8))

plt.subplots_adjust(bottom=0.12,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.grid(True,linestyle=':',color='0.75')
ax1.set_axisbelow(True)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.grid(True,linestyle=':',color='0.75')
ax2.set_axisbelow(True)

plt.xlabel(r'Normalized Frequency $\left(\Omega = \frac{\omega}{\omega_n}\right)$',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.xticks([0,1],['0','$\Omega = 1$'])

# Magnitude plot
ax1.set_ylabel(r'$ k|G(\Omega)| $',family='serif',fontsize=22,weight='bold',labelpad=40)
ax1.plot(wun, k*mag_normal_un, linewidth=2, label=r'$\zeta$ = 0.0')
ax1.plot(w, k*mag_normal_0p1, linewidth=2, linestyle = '-.', label=r'$\zeta$ = 0.1')
ax1.plot(w, k*mag_normal_0p2, linewidth=2, linestyle = ':', label=r'$\zeta$ = 0.2')
ax1.plot(w, k*mag_normal_0p4, linewidth=2, linestyle = '--',label=r'$\zeta$ = 0.4')
ax1.set_ylim(0.0,7.0)
ax1.set_yticks([0,1,2,3,4,5],['0', '1'])

ax1.leg = ax1.legend(loc='upper right', fancybox=True)
ltext  = ax1.leg.get_texts()
plt.setp(ltext,family='Serif',fontsize=16)

# Phase plot 
ax2.set_ylabel(r'$ \phi $ (deg)',family='serif',fontsize=22,weight='bold',labelpad=10)
# ax2.plot(wnorm,TFnorm_phase*180/np.pi,linewidth=2)
ax2.plot(wun, phase_un, linewidth=2, label=r'$\zeta$ = 0.0')
ax2.plot(w, phase_0p1, linewidth=2, linestyle = '-.', label=r'$\zeta$ = 0.1')
ax2.plot(w, phase_0p2, linewidth=2, linestyle = ':', label=r'$\zeta$ = 0.2')
ax2.plot(w, phase_0p4, linewidth=2, linestyle = '--', label=r'$\zeta$ = 0.4')
ax2.set_ylim(-200.0,20.0,)
ax2.set_yticks([0, -90, -180])

ax2.leg = ax2.legend(loc='upper right', fancybox=True)
ltext  = ax2.leg.get_texts()
plt.setp(ltext,family='Serif',fontsize=16)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MassSpring_SeismicTF.pdf',dpi=300)

fig.set_size_inches(9,9) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

