import numpy as np
import control
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

m = 1 
b = 1
Ti = 0.5

num = [Ti, 1]
den = [m * Ti, b * Ti, 0, 0]

sys = control.tf(num, den)

poles, zeros = control.pzmap.pzmap(sys, Plot=False)

# Let's plot the pole-zero map with better formatting than the stock command would

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_position('zero')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('right')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('$\sigma$', family='serif', fontsize=22, weight='bold', labelpad=5)
ax.xaxis.set_label_coords(1.08, .54)


plt.ylabel('$j\omega$', family='serif', fontsize=22, weight='bold', rotation=0, labelpad=10)
ax.yaxis.set_label_coords(3/4, 1.05)


plt.plot(np.real(poles), np.imag(poles), linestyle='', 
         marker='x', markersize=10, markeredgewidth=5, 
         zorder = 10, label=r'Poles')

plt.plot(np.real(zeros), np.imag(zeros), linestyle='', 
         marker='o', markersize=10, markeredgewidth=3, markerfacecolor='none', 
         zorder = 10, label=r'Zeros')


# uncomment below and set limits if needed
plt.xlim(-3, 1)
# plt.ylim(0, 10)

plt.xticks([-3,-2,-1,0,1],['','','',''], bbox=dict(facecolor='black', edgecolor='None', alpha=0.65 ))
plt.yticks([-1.5, -1.0, -0.5, 0, 0.5, 1, 1.5],['','', '', '0', '','',''])

# Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_ExtraZero_PoleZeroMap.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

roots, gains = control.root_locus(sys, kvect=np.linspace(0,100,10001), Plot=False)

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_position('zero')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('right')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
# Define the X and Y axis labels
plt.xlabel('$\sigma$', family='serif', fontsize=22, weight='bold', labelpad=5)
ax.xaxis.set_label_coords(1.08, .54)


plt.ylabel('$j\omega$', family='serif', fontsize=22, weight='bold', rotation=0, labelpad=10)
ax.yaxis.set_label_coords(3/4, 1.05)

plt.plot(np.real(poles), np.imag(poles), linestyle='', 
         marker='x', markersize=10, markeredgewidth=5, 
         zorder = 10, label=r'Poles')

plt.plot(np.real(zeros), np.imag(zeros), linestyle='', 
         marker='o', markersize=10, markeredgewidth=3, markerfacecolor='none', 
         zorder = 10, label=r'Zeros')

plt.plot(roots.real, roots.imag, linewidth=2, linestyle='-', label=r'Data 1')
# plt.plot(x2, y2, linewidth=2, linestyle='--', label=r'Data 2')
# plt.plot(x3, y3, linewidth=2, linestyle='-.', label=r'Data 3')
# plt.plot(x4, y4, linewidth=2, linestyle=':', label=r'Data 4')

plt.xticks([-3,-2,-1,0,1],['','','',''], bbox=dict(facecolor='black', edgecolor='None', alpha=0.65 ))
plt.yticks([-2.0, -1, 0, 1, 2],['','', '0', '',''])

# uncomment below and set limits if needed
plt.xlim(-3, 1)
plt.ylim(-2, 2)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('example_rootLocus_unstable.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

m = 1 
b = 1
Ti = 2.0

num = [Ti, 1]
den = [m * Ti, b * Ti, 0, 0]

sys = control.tf(num, den)

poles, zeros = control.pzmap.pzmap(sys, Plot=False)

# Let's plot the pole-zero map with better formatting than the stock command would

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_position('zero')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('right')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('$\sigma$', family='serif', fontsize=22, weight='bold', labelpad=5)
ax.xaxis.set_label_coords(1.08, .54)


plt.ylabel('$j\omega$', family='serif', fontsize=22, weight='bold', rotation=0, labelpad=10)
ax.yaxis.set_label_coords(3/4, 1.05)


plt.plot(np.real(poles), np.imag(poles), linestyle='', 
         marker='x', markersize=10, markeredgewidth=5, 
         zorder = 10, label=r'Poles')

plt.plot(np.real(zeros), np.imag(zeros), linestyle='', 
         marker='o', markersize=10, markeredgewidth=3, markerfacecolor='none', 
         zorder = 10, label=r'Zeros')


# uncomment below and set limits if needed
plt.xlim(-3, 1)
# plt.ylim(0, 10)

plt.xticks([-3,-2,-1,0,1],['','','',''], bbox=dict(facecolor='black', edgecolor='None', alpha=0.65 ))
plt.yticks([-1.5, -1.0, -0.5, 0, 0.5, 1, 1.5],['','', '', '0', '','',''])

# Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_ExtraZero_PoleZeroMap.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

roots, gains = control.root_locus(sys, kvect=np.linspace(0,100,10001), Plot=False)

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_position('zero')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('right')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
# Define the X and Y axis labels
plt.xlabel('$\sigma$', family='serif', fontsize=22, weight='bold', labelpad=5)
ax.xaxis.set_label_coords(1.08, .54)


plt.ylabel('$j\omega$', family='serif', fontsize=22, weight='bold', rotation=0, labelpad=10)
ax.yaxis.set_label_coords(3/4, 1.05)

plt.plot(np.real(poles), np.imag(poles), linestyle='', 
         marker='x', markersize=10, markeredgewidth=5, 
         zorder = 10, label=r'Poles')

plt.plot(np.real(zeros), np.imag(zeros), linestyle='', 
         marker='o', markersize=10, markeredgewidth=3, markerfacecolor='none', 
         zorder = 10, label=r'Zeros')

plt.plot(roots.real, roots.imag, linewidth=2, linestyle='-', label=r'Data 1')
# plt.plot(x2, y2, linewidth=2, linestyle='--', label=r'Data 2')
# plt.plot(x3, y3, linewidth=2, linestyle='-.', label=r'Data 3')
# plt.plot(x4, y4, linewidth=2, linestyle=':', label=r'Data 4')

plt.xticks([-3,-2,-1,0,1],['','','',''], bbox=dict(facecolor='black', edgecolor='None', alpha=0.65 ))
plt.yticks([-2.0, -1, 0, 1, 2],['','', '0', '',''])

# uncomment below and set limits if needed
plt.xlim(-3, 1)
plt.ylim(-2, 2)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('example_rootLocus_unstable.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# To find the root for a particular gain, we'll find the index of 
# the gains vector that most closely matches the desired gain.
# Then, we'll get the pol
kp0p1_roots = roots[np.argmin(np.abs(0.1-gains))]
kp1_roots = roots[np.argmin(np.abs(1-gains))]
kp5_roots = roots[np.argmin(np.abs(5-gains))]

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_position('zero')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('right')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
# Define the X and Y axis labels
plt.xlabel('$\sigma$', family='serif', fontsize=22, weight='bold', labelpad=5)
ax.xaxis.set_label_coords(1.08, .54)


plt.ylabel('$j\omega$', family='serif', fontsize=22, weight='bold', rotation=0, labelpad=10)
ax.yaxis.set_label_coords(2/3, 1.05)

plt.plot(np.real(poles), np.imag(poles), linestyle='', 
         marker='x', markersize=10, markeredgewidth=5, 
         zorder = 10, label=r'')

plt.plot(np.real(zeros), np.imag(zeros), linestyle='', 
         marker='o', markersize=10, markeredgewidth=3, markerfacecolor='none', 
         zorder = 10, label=r'')

plt.plot(roots.real, roots.imag, linewidth=2, linestyle='-', label=r'')
plt.plot(kp0p1_roots.real, kp0p1_roots.imag, linewidth=2, marker='o', markersize=10, linestyle='', label=r'$k_p = 0.1$')
plt.plot(kp1_roots.real, kp1_roots.imag, linewidth=2, marker='*', markersize=10, linestyle='', label=r'$k_p = 1.0$')
plt.plot(kp5_roots.real, kp5_roots.imag, linewidth=2, marker='d', markersize=10, linestyle='', label=r'$k_p = 5.0$')

plt.xticks([-2,-1,0,1],['','','',''], bbox=dict(facecolor='black', edgecolor='None', alpha=0.65 ))
plt.yticks([-3,-2.0, -1, 0, 1, 2,3],['','','', '0', '','',''])

# uncomment below and set limits if needed
plt.xlim(-2, 1)
plt.ylim(-3, 3)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('example_rootLocus_unstable.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

sys_kp0p1 = control.feedback(0.1 * sys)
sys_kp1 = control.feedback(1.0 * sys)
sys_kp5 = control.feedback(5.0 * sys)

# Time vector to simulate the step command on. The system is slow, so we need a long time.
time = np.linspace(0, 30, 3001)  

resp_kp0p1, t_kp0p1 = control.step(sys_kp0p1, time)
resp_kp1, t_kp1 = control.step(sys_kp1, time)
resp_kp5, t_kp5 = control.step(sys_kp5, time)

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
plt.ylabel('Position (m)', family='serif', fontsize=22, weight='bold', labelpad=10)

plt.plot(t_kp0p1, resp_kp0p1 , linewidth=2, linestyle='-', label=r'$k_p = 0.1$')
plt.plot(t_kp1, resp_kp1, linewidth=2, linestyle='--', label=r'$k_p = 1.0$')
plt.plot(t_kp5, resp_kp5, linewidth=2, linestyle='-.', label=r'$k_p = 5.0$')
# plt.plot(x4, y4, linewidth=2, linestyle=':', label=r'Data 4')

# uncomment below and set limits if needed
plt.xlim(0, 30)
plt.ylim(0, 2)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('plot_filename.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# This cell will just improve the styling of the notebook
from IPython.core.display import HTML
import urllib.request
response = urllib.request.urlopen("https://cl.ly/1B1y452Z1d35")
HTML(response.read().decode("utf-8"))

