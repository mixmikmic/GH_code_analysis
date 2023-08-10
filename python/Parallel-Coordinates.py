import pandas as pd
import numpy as np
import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import scatter_matrix

iris_data = pd.read_csv('Parallel-Coordinates_Iris-dataset.csv')
iris_data.head()

iris_data.tail()

# Remove the 'ID' column from Iris flower dataset
iris = iris_data.drop('Id', 1)
iris.head()

# Get columns list of Iris flower dataset
cols = iris.columns.tolist()
print(cols)

# Permute randomly the columns list of Iris flower dataset
cols_new= np.random.permutation(cols)
cols_new= cols_new.tolist()
print (cols_new)

# Create a new version of Iris flower dataset with permutated columns
iris_new = iris[cols_new]
iris_new.head()

# Plot two parallel coordinates of Iris flower dataset with different ordering of the axis.
fig = plt.figure(figsize=(15,6))

ax1 = plt.subplot(121)
parallel_coordinates(iris, 'Species',color=['r','g','b'])


ax2= plt.subplot(122)
parallel_coordinates(iris_new, 'Species',color=['r','g','b'])


ax1.spines["left"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["top"].set_visible(False)
#ax1.get_yaxis().tick_left() # remove unneeded ticks 

ax2.spines["left"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
#ax2.get_yaxis().tick_left() # remove unneeded ticks 

plt.tight_layout()
plt.show()

scatter_matrix(iris, figsize=(10, 8),c=['r','g','b'],diagonal='kde')
plt.show()

#vectors to plot
y1=[1,2,1]
y2=[4,2,3]
y3=[4,2,1]
y4=[1,2,3]

#spines
x=[1,2,3]

plt.figure(1)
fig,(ax1,ax2) = plt.subplots(1,2, sharey=False)
ax3 = fig.add_axes([1, 0.125, 0.4, 0.775], label='axes1')
ax4 = fig.add_axes([1.4, 0.125, 0.4, 0.775], label='axes1')


#plot vectors y1 and y2 without color coding
ax1.plot(x,y1,'black', x,y2,'black')
ax2.plot(x,y1,'black', x,y2,'black')
ax1.set_xlim([ x[0],x[1]])
ax2.set_xlim([ x[1],x[2]])
ax1.spines["bottom"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.set_xticks([]) #Remove unneeded ticks
ax1.get_yaxis().tick_left() 
ax2.set_xticks([]) 
ax2.set_yticks([]) 
plt.subplots_adjust(wspace=0)


#plot vectors y3 and y4 withour color coding
ax3.plot(x,y3,'black', x,y4,'black')
ax4.plot(x,y3,'black', x,y4,'black')
ax3.set_xlim([ x[0],x[1]])
ax4.set_xlim([ x[1],x[2]])
ax3.spines["bottom"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax4.spines["bottom"].set_visible(False)
ax4.spines["top"].set_visible(False)
ax3.set_xticks([]) #Remove unneeded ticks
ax3.get_yaxis().tick_left() 
ax4.set_xticks([]) 
ax4.set_yticks([]) 

fig.suptitle('Single color lines', x=1,y=1.05,fontsize=14, fontweight='bold')
ax1.set_title("A (y1=[1,2,1],y2=[4,2,3])")
ax3.set_title("B (y3=[4,2,1],y4=[1,2,3])")

#################################################

plt.figure(2)
fig,(ax1,ax2) = plt.subplots(1,2, sharey=False)
ax3 = fig.add_axes([1, 0.125, 0.4, 0.775], label='axes1')
ax4 = fig.add_axes([1.4, 0.125, 0.4, 0.775], label='axes1')

#plot vectors y1 and y2 with color coding
ax1.plot(x,y1,'red', x,y2,'blue')
ax2.plot(x,y1,'red', x,y2,'blue')
ax1.set_xlim([ x[0],x[1]])
ax2.set_xlim([ x[1],x[2]])
ax1.spines["bottom"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.set_xticks([]) #Remove unneeded ticks
ax1.get_yaxis().tick_left() 
ax2.set_xticks([]) 
ax2.set_yticks([]) 
plt.subplots_adjust(wspace=0)

#plot vectors y3 and y4 with color coding
ax3.plot(x,y3,'red', x,y4,'blue')
ax4.plot(x,y3,'red', x,y4,'blue')
ax3.set_xlim([ x[0],x[1]])
ax4.set_xlim([ x[1],x[2]])
ax3.spines["bottom"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax4.spines["bottom"].set_visible(False)
ax4.spines["top"].set_visible(False)
ax3.set_xticks([]) #Remove unneeded ticks
ax3.get_yaxis().tick_left() 
ax4.set_xticks([]) 
ax4.set_yticks([]) 

fig.suptitle('Color-coded lines', x=1,y=1.05,fontsize=14, fontweight='bold')
ax1.set_title("C (y1=[1,2,1], y2=[4,2,3])")
ax3.set_title("D (y3=[4,2,1], y4=[1,2,3])")

plt.show()



