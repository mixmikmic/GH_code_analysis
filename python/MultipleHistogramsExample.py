#Here we load all the necessary modules to make the plots, as well as simulate some data.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
y1 = np.random.randn(10000)
y2 = np.random.randn(10000)+1
print 'y1: ',y1
print 'y2: ',y2

# Basic Histogram Plot Syntax
plt.hist(y1)
plt.show()

# Plot Multiple Histograms with legend

# Plot the first histogram
plt.hist(y1,
         bins=20,
         alpha=0.8,         # Opaqueness
         label='Group 1',   # Legend Label
         facecolor='green'
         )

# Overplot the second histogram
plt.hist(y2,
         bins=20,
         alpha=0.8,         # Opaqueness
         label='Group 2',   # Legend Label
         facecolor='pink'
         )

# Label your plot
plt.title('Main Plot Title',fontsize=25)
plt.ylabel('Count',fontsize=20)
plt.yticks(fontsize=15)
plt.xlabel('X Axis Label',fontsize=20)
plt.xticks(fontsize=15)

# Add the legend
plt.legend()

plt.show()



