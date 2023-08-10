import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# simplest plot:  list of values
plt.plot([1,3,2,4,3,5,4,6,5,7]) # plot a list of values

# set up a few things to plot
a = np.array((1,2,3,4,5))
b = np.array((6,7,5,9,13))
c = np.random.rand(10,10) # 10x10 array with random numbers between 0 and 1

plt.plot(a, b, marker='o', color='red')
plt.ylabel('y label')
plt.xlabel('x label')
plt.title('Title')

plt.contour(c)
#plt.contourf(c)
plt.colorbar(label='colorbar label')

plt.contourf(c, cmap='YlOrRd')
plt.colorbar(label='colorbar label')

plt.plot(a, b, lw=3, color='dodgerblue')
plt.plot(a, b-2, lw=3, color='firebrick')
plt.plot(a+1, b-3, linewidth=3, linestyle='--', color='0.25', marker='s')

plt.grid()

plt.title('Figure')
plt.xlabel('x axis label')
plt.ylabel('y axis label')

#plt.xlim(-5,10)
#plt.ylim(0,20)

# 1
fig = plt.figure()

# 2
ax = fig.add_subplot(1,1,1) # (rows, columns, index)

# 3
#ax.plot(a, b, color='orangered', lw=3)

# 4
ax.grid()
ax.set_title('Title', fontsize=16)
ax.set_xlabel('x label', fontsize=16)
ax.set_ylabel('y label', fontsize=16)
ax.tick_params(labelsize=16)

fig = plt.figure()
ax = fig.add_subplot(1,1,1) # (rows, columns, index)
ax_plot = ax.pcolormesh(c, cmap='Reds')
plt.colorbar(ax_plot, label='units')

ax.set_xlabel('x label')

my_fontsize = 16

fig = plt.figure(figsize=(6,5))

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(212) # commas not necessary if numbers are single digits

ax1.plot(a, b, label='b line') # add labels to lines to create legend
ax1.plot(a, b+3, label='b+3 line')
ax1.legend(fontsize=my_fontsize)

ax2.contourf(c)

ax1.grid()
ax1.set_title('Axes 1 title', fontsize=my_fontsize)
ax2.set_title('Axes 2 title', fontsize=my_fontsize)

ax1.tick_params(labelsize=my_fontsize)
ax2.tick_params(labelsize=my_fontsize)

fig.tight_layout()

my_fontsize = 12

fig, ax = plt.subplots(1,1)

fig.set_size_inches(4,3)
ax.plot(a, b, c='dodgerblue') # first figure

left, bottom, width, height = 0.2,0.45,0.3,0.35
ax_inset = fig.add_axes([left,bottom,width,height]) # create inset axes
ax_inset.scatter(a, b, c='orangered') # add a plot to inset

ax.tick_params(labelsize=my_fontsize)
ax_inset.tick_params(labelsize=my_fontsize)

fig.savefig('figure_example.png', dpi=300, bbox_inches='tight')

