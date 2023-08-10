import numpy as np              # Grab all of the NumPy functions with nickname np

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
import matplotlib.pyplot as plt

# Define the system pararmeters
k = 2*(2*np.pi)**2            # Spring constant (N/m)
m1 = 1.75                     # Sprung/main mass (kg)
m2 = 0.25                     # rotating mass (kg)
b = m2/(m1 + m2)              # mass ratio    
wn = np.sqrt(k/(m1 + m2))     # natural frequency (rad/s)
l = 0.1                       # Eccentricity


z = 0.1                       # Damping Ratio
c = 2*z*wn*(m1 + m2)          # Select c based on desired amping ratio

# Set up the frequency range
w = np.linspace(0,5*wn,2000)            # Freq range, 0-5*wn with 2000 points in-between

# Look at undamped case
z = 0.0
x_mag_un = (l*b*w**2)/np.sqrt((wn**2-w**2)**2+(2*z*w*wn)**2)

# Look at z=0.1
z = 0.1
x_mag_0p1 = (l*b*w**2)/np.sqrt((wn**2-w**2)**2+(2*z*w*wn)**2)

# Look at z=0.2
z = 0.2
x_mag_0p2 = (l*b*w**2)/np.sqrt((wn**2-w**2)**2+(2*z*w*wn)**2)

# Look at z=0.4
z = 0.4
x_mag_0p4 = (l*b*w**2)/np.sqrt((wn**2-w**2)**2+(2*z*w*wn)**2)


w = w/wn # Scale frequency so the plot is normalized by the natural frequency

# Let's plot the magnitude of the frequency response

fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='Serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='Serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.95')
ax.set_axisbelow(True)

plt.xlabel(r'Normalized Frequency $(\Omega)$',family='Serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Normalized Amp., $\frac{|x|}{e \beta}$',family='Serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(w,x_mag_un/(l*b),  linewidth=2, linestyle = ':',  label=r'$\zeta = 0.0$')
plt.plot(w,x_mag_0p1/(l*b), linewidth=2, linestyle = '-',  label=r'$\zeta = 0.1$')
plt.plot(w,x_mag_0p2/(l*b), linewidth=2, linestyle = '-.', label=r'$\zeta = 0.2$')
plt.plot(w,x_mag_0p4/(l*b), linewidth=2, linestyle = '--', label=r'$\zeta = 0.4$')

plt.xlim(0,5)
plt.ylim(0,7)

leg = plt.legend(loc='upper right', fancybox=True)
ltext  = leg.get_texts() 
plt.setp(ltext,family='Serif',fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)


# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('RotatingImbalance_Freq_Resp_mag.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

