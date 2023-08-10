get_ipython().magic('matplotlib notebook')

import numpy as np
import matplotlib.pyplot as plt 

# Create the X data points as a numpy array 
X = np.linspace(-10, 10, 255)

# Compute two quadratic functions 
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 

plt.plot(X, Y1)
plt.plot(X, Y2)

# Create a new figure of size 8x6 points, using 72 dots per inch 
plt.figure(figsize=(8,6), dpi=72)

# Create a new subplot from a 1x1 grid 
plt.subplot(111)

# Create the data to plot 
X = np.linspace(-10, 10, 255)
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 

# Plot the first quadratic using a blue color with a continuous line of 1px
plt.plot(X, Y1, color='blue', linewidth=1.0, linestyle='-')

# Plot the second quadratic using a green color with a continuous line of 1px
plt.plot(X, Y2, color='green', linewidth=1.0, linestyle='-')

# Set the X limits 
plt.xlim(-10, 10)

# Set the X ticks 
plt.xticks(np.linspace(-10, 10, 9, endpoint=True))

# Set the Y limits 
plt.ylim(0, 350)

# Set the Y ticks 
plt.yticks(np.linspace(0, 350, 5, endpoint=True))

# Save the figure to disk 
plt.savefig("figures/quadratics.png")

# Create the data to plot 
# This data will be referenced for the next plots below
# For Jupyter notebooks, pay attention to variables! 

X = np.linspace(-10, 10, 255)
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 

from matplotlib.colors import ListedColormap

colors = 'bgrmyck'
fig, ax = plt.subplots(1, 1, figsize=(7, 1))
ax.imshow(np.arange(7).reshape(1,7), cmap=ListedColormap(list(colors)), interpolation="nearest", aspect="auto")
ax.set_xticks(np.arange(7) - .5)
ax.set_yticks([-0.5,0.5])
ax.set_xticklabels([])
ax.set_yticklabels([])

# plt.style.use('fivethirtyeight')

# Note that I'm going to use temporary styling so I don't mess up the notebook! 
with plt.style.context(('fivethirtyeight')):
    plt.plot(X, Y1)
    plt.plot(X, Y2)

# To see the available styles:
for style in plt.style.available: print("- {}".format(style))

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-")

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(Y1.min()*-1.1, Y2.max()*1.1)

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-", label="Y1")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-", label="Y2")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(Y1.min()*-1.1, Y2.max()*1.1)

plt.title("Two Quadratic Curves")
plt.legend(loc='best')

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-", label="Y1")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-", label="Y2")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(0, Y2.max()*1.1)

plt.title("Two Quadratic Curves")
plt.legend(loc='best')

# Annotate the blue line 
x = 6 
y = 2*x ** 2 + 10
plt.plot([x,x], [0, y], color='blue', linewidth=1.5, linestyle='--')
plt.scatter([x,], [y,], color='blue', s=50, marker='D')

plt.annotate(
    r'$2x^2+10={}$'.format(y), xy=(x,y), xycoords='data', xytext=(10,-50), 
    fontsize=16, textcoords='offset points',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
)

# Annotate the red line
x = -3
y = 3*x ** 2 + 50
plt.plot([x,x], [0, y], color='red', linewidth=1.5, linestyle='--')
plt.scatter([x,], [y,], color='red', s=50, marker='s')

plt.annotate(
    r'$3x^2+50={}$'.format(y), xy=(x,y), xycoords='data', xytext=(10,50), 
    fontsize=16, textcoords='offset points',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
)

# Make up some fake data
x = np.linspace(-np.pi, np.pi, 50) 
y = np.linspace(-np.pi, np.pi, 50)
X,Y = np.meshgrid(x,y)
Z = np.sin(X + Y/4)

fig = plt.figure(figsize = (12,2.5))
fig.subplots_adjust(wspace=0.3)

# Blues
plt.subplot(1,3,1)
plt.pcolormesh(X, Y, Z, cmap=plt.cm.get_cmap('Blues'))
plt.colorbar()
plt.axis([-3, 3, -3, 3])
plt.title('Sequential')

# Red-Blue
plt.subplot(1,3,2)
plt.pcolormesh(X, Y, Z, cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.axis([-3, 3, -3, 3])
plt.title('Diverging')

# Red-Blue
plt.subplot(1,3,3)
plt.pcolormesh(X, Y, Z, cmap=plt.cm.get_cmap('plasma'))
plt.colorbar()
plt.axis([-3, 3, -3, 3])
plt.title('Fancy!')

