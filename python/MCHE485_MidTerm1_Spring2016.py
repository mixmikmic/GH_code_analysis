import numpy as np                  # import NumPy with the namespace np
from scipy.integrate import odeint  # import the ODE solver for simluation

get_ipython().run_line_magic('matplotlib', 'inline')

# import the plotting functions
import matplotlib.pyplot as plt

# define the system paramters
m = 1.0          # mass (kg)
k1 = 5.0         # k1 spring constant (N/m)
k2 = 5.0         # k2 spring constant (N/m)

# define the natural frequency (rad/s)
wn = (k1 * k2) / (m * (k1 + k2))  

def eq_of_motion(w, t, p):
    """
    Defines the differential equations for the mass-spring-damper system.

    Arguments:
        w :  vector of the state variables:
        t :  time
        p :  vector of the parameters:
            wn = natural frequency (rad/s)
            zeta = damping ratio
    """
    x, x_dot = w
    wn, zeta = p

    # Create sysODE = (x', x_dot')
    sysODE = [x_dot,
              -wn**2 * x - 2 * zeta * wn * x_dot]
    return sysODE

# Set up simulation parameters

# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
stoptime = 10.0
numpoints = 10001

# Create the time samples for the output of the ODE solver
t = np.linspace(0.0, stoptime, numpoints)

# Initial conditions
x_init = 1.0                        # initial position
x_dot_init = 2.0                    # initial velocity

# Pack the initial conditions into a list
x0 = [x_init, x_dot_init]

# zeta = 0.0 case
# Define the damping ratio
zeta = 0.0

# Pack the damping ratio and natural frequency
p = [wn, zeta]
resp_zeta0p0 = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)


# zeta = 0.2 case
# Define the damping ratio
zeta = 0.2

# Pack the damping ratio and natural frequency
p = [wn, zeta]
resp_zeta0p2 = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)


# zeta = 0.7 case
# Define the damping ratio
zeta = 0.7

# Pack the damping ratio and natural frequency
p = [wn, zeta]
resp_zeta0p7 = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Position (m)', family='serif', fontsize=22, weight='bold', labelpad=5)

plt.plot(t, resp_zeta0p0[:,0], linewidth=2, linestyle='-', label=r'$\zeta = 0.0$')
plt.plot(t, resp_zeta0p2[:,0], linewidth=2, linestyle='--', label=r'$\zeta = 0.2$')
plt.plot(t, resp_zeta0p7[:,0], linewidth=2, linestyle='-.', label=r'$\zeta = 0.7$')

plt.annotate('$v_0$',
         xy=(0.0, 1), xycoords='data',
         xytext=(0.5, 2), textcoords='data', fontsize=20,
         arrowprops=dict(arrowstyle='<|-', linewidth = 2, color="#984ea3"), color = "#984ea3")



# uncomment below and set limits if needed
# plt.xlim(0, 5)
plt.ylim(-1.5, 2.5)
plt.yticks([-1, 0, 1], [r'$-x_0$', '0', '$x_0$'])

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE485_Midterm1_Prob1d_Spring2016.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# Define the system parameters
m = 1.0    # mass (kg)
k = 2.0    # spring constant (N/m)
l = 1.0    # link length (m)
g = 9.81   # gravity (m/s^2)

# Now, define the natural frequency
wn = np.sqrt((m * g + k * l) / (m * l))

# Set up input parameters
w = np.linspace(0.01,3*wn,1000)   # Frequency range for freq response plot, 0-3x wn with 1000 points in-between

# Define the transfer function
TF_amp = (k / (m*l)) / (wn**2 - w**2)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(TF_amp) >= 25)
TF_amp[pos] = np.nan
w[pos] = np.nan

# Now define the magnitude and phase of this TF
TF_mag = np.sqrt(TF_amp**2)
TF_phase = -np.arctan2(0, TF_amp)

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

plt.xlabel(r'Input Frequency ($\omega$)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.xticks([0, wn], ['', '$\omega = \omega_n$'])

# Magnitude plot
ax1.set_ylabel(r'$ |G(\omega)| $', family='serif', fontsize=22, weight='bold', labelpad=40)
ax1.plot(w, TF_mag, linewidth=2)

ax1.set_ylim(0.0, 5.0)
ax1.set_yticks([0, 1, 2, 3, 4, 5])
ax1.set_yticklabels(['$0$', '$1$', '', '', '', ''])
plt.setp(ax1.get_ymajorticklabels(),family='serif', fontsize=18, weight = 'light')

# Phase plot 
ax2.set_ylabel(r'$ \phi $ (deg)',family='serif',fontsize=22,weight='bold',labelpad=10)
ax2.plot(w, TF_phase * 180/np.pi, linewidth=2)
ax2.set_ylim(-200.0, 20.0,)
ax2.set_yticks([0, -90, -180])

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
plt.savefig('MCHE485_Midterm1_Prob2f_Spring2016', dpi=300)

fig.set_size_inches(9,9) # Resize the figure for better display in the notebook

# Define the system parameters
m = 1.0              # mass (kg)
k = 2.0              # spring constant (N/m)
c = 1.0              # damping coefficient (Ns/m)
l = 2.0              # total link length (m)
l1 = 1.0             # length to spring connection (m)
l2 = 1.5             # length to damper connection (m)

Io = 1/3 * m * l**2  # The rod moment of intertia about the pin joint

wn = np.sqrt(k*l1**2 / Io)  # Define the natural frequency

# Define the damping ratio
zeta = (c * l2**2 / Io) / (2 * wn)

# Set up input parameters
w = np.linspace(0.01, 3*wn, 1000)   # Frequency range for freq response plot, 0-3x wn with 1000 points in-between
   
# The undamped case
zeta = 0
TF_mag_0p0 = 1 / (Io * np.sqrt((wn**2 - w**2)**2 + (2 * zeta * w * wn)**2))
TF_phase_0p0 = -np.arctan2(2 * zeta * w * wn, wn**2 - w**2)

# Let's mask the discontinuity in the undamped case, so it isn't plotted
pos = np.where(np.abs(TF_mag_0p0) >= 25)
TF_mag_0p0[pos] = np.nan
TF_phase_0p0[pos] = np.nan
# w[pos] = np.nan

# The zeta = 0.2 case
zeta = 0.2
TF_mag_0p2 = 1 / (Io * np.sqrt((wn**2 - w**2)**2 + (2 * zeta * w * wn)**2))
TF_phase_0p2 = -np.arctan2(2 * zeta * w * wn, wn**2 - w**2)

# The zeta = 0.7 case
zeta = 0.7
TF_mag_0p7 = 1 / (Io * np.sqrt((wn**2 - w**2)**2 + (2 * zeta * w * wn)**2))
TF_phase_0p7 = -np.arctan2(2 * zeta * w * wn, wn**2 - w**2)

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

plt.xlabel(r'Input Frequency ($\omega$)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.xticks([0, wn], ['', '$\omega = \omega_n$'])

# Magnitude plot
ax1.set_ylabel(r'$ |G(\omega)| $', family='serif', fontsize=22, weight='bold', labelpad=40)
ax1.plot(w, TF_mag_0p0, linewidth=2, linestyle = '-', label = r'$\zeta = 0.0$')
ax1.plot(w, TF_mag_0p2, linewidth=2, linestyle = '--', label = r'$\zeta = 0.2$')
ax1.plot(w, TF_mag_0p7, linewidth=2, linestyle = ':', label = r'$\zeta = 0.7$')

ax1.set_ylim(0.0, 5.0)
ax1.set_yticks([0, 1, 2, 3, 4, 5])
ax1.set_yticklabels(['$0$', '', '', '', '', ''])
plt.setp(ax1.get_ymajorticklabels(),family='serif', fontsize=18, weight = 'light')


# Create the legend, then fix the fontsize
leg = ax1.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)



# Phase plot 
ax2.set_ylabel(r'$ \phi $ (deg)',family='serif',fontsize=22,weight='bold',labelpad=10)
ax2.plot(w, TF_phase_0p0 * 180/np.pi, linewidth=2, linestyle = '-', label = r'$\zeta = 0.0$')
ax2.plot(w, TF_phase_0p2 * 180/np.pi, linewidth=2, linestyle = '--', label = r'$\zeta = 0.2$')
ax2.plot(w, TF_phase_0p7 * 180/np.pi, linewidth=2, linestyle = ':', label = r'$\zeta = 0.7$')

ax2.set_ylim(-200.0, 20.0,)
ax2.set_yticks([0, -90, -180])


# Create the legend, then fix the fontsize
leg = ax2.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)



# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MCHE485_Midterm1_Prob3d_Spring2016', dpi=300)

fig.set_size_inches(9,9) # Resize the figure for better display in the notebook

# Define the system parameters
m = 1.0         # mass (kg)
k = 5.0         # spring constant (N/m)

# define the natural frequency (rad/s)
wn = np.sqrt((2 * k) / m)

# Set up input parameters
w = np.linspace(0.01, 3 * wn, 1000)   # Frequency range for freq response plot, 0-3x wn with 1000 points in-between


# Now, define the parameters and TF magnitude and phase for each damping ratio

# The undamped case
zeta = 0
TF_mag_0p0 = np.sqrt(wn**4 + (2 * zeta * w * wn)**2) / np.sqrt((wn**2 - w**2)**2 + (2 * zeta * w * wn)**2)
TF_phase_0p0 = np.arctan2(2 * zeta * w, wn) - np.arctan2(2 * zeta * w * wn, wn**2 - w**2)

# Let's mask the discontinuity in the undamped case, so it isn't plotted
pos = np.where(np.abs(TF_mag_0p0) >= 25)
TF_mag_0p0[pos] = np.nan
TF_phase_0p0[pos] = np.nan
# w[pos] = np.nan

# The zeta = 0.2 case
zeta = 0.2
TF_mag_0p2 = np.sqrt(wn**4 + (2 * zeta * w * wn)**2) / np.sqrt((wn**2 - w**2)**2 + (2 * zeta * w * wn)**2)
TF_phase_0p2 = np.arctan2(2 * zeta * w, wn) - np.arctan2(2 * zeta * w * wn, wn**2 - w**2)

# The zeta = 0.7 case
zeta = 0.7
TF_mag_0p7 = np.sqrt(wn**4 + (2 * zeta * w * wn)**2) / np.sqrt((wn**2 - w**2)**2 + (2 * zeta * w * wn)**2)
TF_phase_0p7 = np.arctan2(2 * zeta * w, wn) - np.arctan2(2 * zeta * w * wn, wn**2 - w**2)

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

plt.xlabel(r'Input Frequency ($\omega$)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.xticks([0, wn], ['', '$\omega = \omega_n$'])

# Magnitude plot
ax1.set_ylabel(r'$ |G(\omega)| $', family='serif', fontsize=22, weight='bold', labelpad=40)
ax1.plot(w, TF_mag_0p0, linewidth=2, linestyle = '-', label = r'$\zeta = 0.0$')
ax1.plot(w, TF_mag_0p2, linewidth=2, linestyle = '--', label = r'$\zeta = 0.2$')
ax1.plot(w, TF_mag_0p7, linewidth=2, linestyle = ':', label = r'$\zeta = 0.7$')

ax1.set_ylim(0.0, 5.0)
ax1.set_yticks([0, 1, 2, 3, 4, 5])
ax1.set_yticklabels(['$0$', '', '', '', '', ''])
plt.setp(ax1.get_ymajorticklabels(),family='serif', fontsize=18, weight = 'light')


# Create the legend, then fix the fontsize
leg = ax1.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)



# Phase plot 
ax2.set_ylabel(r'$ \phi $ (deg)',family='serif',fontsize=22,weight='bold',labelpad=10)
ax2.plot(w, TF_phase_0p0 * 180/np.pi, linewidth=2, linestyle = '-', label = r'$\zeta = 0.0$')
ax2.plot(w, TF_phase_0p2 * 180/np.pi, linewidth=2, linestyle = '--', label = r'$\zeta = 0.2$')
ax2.plot(w, TF_phase_0p7 * 180/np.pi, linewidth=2, linestyle = ':', label = r'$\zeta = 0.7$')

ax2.set_ylim(-200.0, 20.0,)
ax2.set_yticks([0, -90, -180])


# Create the legend, then fix the fontsize
leg = ax2.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)



# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('MCHE485_Midterm1_Prob4d_Spring2016', dpi=300)

fig.set_size_inches(9,9) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
import codecs
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(codecs.open(css_file, 'r', 'utf-8').read())

