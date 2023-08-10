import matplotlib
import matplotlib.pyplot as plt
# need to tell IPython to display plots within the notebook
get_ipython().magic('matplotlib inline')
# this is not strictly necessary, it just improves the style of the Matplotlib plots
matplotlib.style.use("ggplot")

# create a new figure
plt.figure()
# set up the plot
plt.plot([1,2,3,5,8])
plt.ylabel("My Numbers")
plt.show()

import numpy as np
# this is our data
xvalues = np.array([ 1, 5, 8, 3, 7 ])
yvalues = np.array([ 0.4, 0.25, 0.65, 0.7, 0.2 ])
# create the figure
plt.figure()
# create a scatter plot
plt.scatter(xvalues,yvalues)
# customise the range of the axes: values are [xmin xmax ymin ymax]
plt.axis([0,10,0,1])
# add labels to the axes
plt.xlabel("My X Numbers")
plt.ylabel("My Y Numbers")

x = np.random.random(15) * 10
y = np.arange(15) + np.random.random(15) + 5 
print(x)
print(y)

# set up the new figure
plt.figure()
# set up the scatter plot
plt.scatter(x, y, c="green", s=30, marker="*") # s is for size
# plt.axis([0,10,0,1])
# customise it
plt.xlabel("X Scores")
plt.ylabel("Y Scores")

import numpy as np
values = np.array( [ 5, 11, 14, 6 ] )
names = [ "Alice", "Paul", "Susan", "Bob" ]

# these are the corresponding positions on the y-axis, for plotting purposes
y_pos = np.arange(len(names)) # np.arrange(4)
y_pos

# create a new figure
plt.figure()
# set up the bar chart
plt.barh(y_pos, values, align='center')
plt.yticks(y_pos, names)
plt.xlabel("Points")
plt.title("My Chart")

# set up the new figure
plt.figure()
# generate and plot 4 groups of points
for color in ["red","green","blue","orange"]:
    # generate 8 random points
    x = np.random.random(8) 
    y = np.random.random(8)
    plt.scatter(x, y, c=color, s=50, label=color)

# some sample data
counts = np.array( [18, 23, 7] )
parties = ["Republicans", "Democrats", "Others"]

# set up the new figure, specifying the size of the figure
plt.figure(figsize=(4, 4))
# create the pie chart on the sample data
p = plt.pie(counts, labels=parties, colors=["red","blue","grey"], autopct="%.1f%%")
# pie = p = plt.pie(counts, labels=parties, colors=["red","blue","grey"], autopct="%d")



