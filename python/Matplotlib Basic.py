# Import the matplot lib library
import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(1,35,5)
y = np.arange(1,10,2)
y = y **2
print ("x",x)
print('y',y)

get_ipython().run_line_magic('matplotlib', 'inline')

# Create a plt using the plot function
plt.plot(x,y)

# Display the plot
plt.show()

x2=np.random.randint(1,33,5)
y2 = np.arange(1,80,16)
print('x2',x2)
print('y2',y2)

# Create a plt using the plot function
#plt.add_axes[0,0,1,1]
plt.plot(x,y,label='Plots Legend Label')
plt.plot(x2,y2,label='Plots Legend Label 2')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Create the legend based on the values specified while cerating the plot 
plt.legend()

# Display the plot
plt.show()

import matplotlib.pyplot as plt

plt.bar(x,y, label="Example one")

plt.bar(x2,y2, label="Example two", color='g')
plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')

plt.title('Epic Graph\nAnother Line! Whoa')
    
plt.show()



