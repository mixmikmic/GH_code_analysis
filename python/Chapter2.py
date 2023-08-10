get_ipython().magic('matplotlib inline')

#P32-35: Simple plot example
x_numbers = [1, 2, 3]
y_numbers = [2, 4, 6]
from pylab import plot, show
plot(x_numbers, y_numbers)
show()
plot(x_numbers, y_numbers, marker='o')
show()
plot(x_numbers, y_numbers, 'o')
show()

#P35: Plot the temperature in New York City
nyc_temp = [53.9, 56.3, 56.4, 53.4, 54.5, 55.8, 56.8, 55.0, 55.3, 54.0,
56.7, 56.4, 57.3]
plot(nyc_temp, marker='o')
show()

#P37: Plot the temperature in New York City with the years on the X-axis
nyc_temp = [53.9, 56.3, 56.4, 53.4, 54.5, 55.8, 56.8, 55.0, 55.3, 54.0,
56.7, 56.4, 57.3]
years = range(2000, 2013)
plot(years, nyc_temp, marker='o')
show()

#P39: Comparing the monthly temperature trends in New York City
nyc_temp_2000 = [31.3, 37.3, 47.2, 51.0, 63.5, 71.3, 72.3, 72.7, 66.0,
57.0, 45.3, 31.1]
nyc_temp_2006 = [40.9, 35.7, 43.1, 55.7, 63.1, 71.0, 77.9, 75.8, 66.6,
56.2, 51.9, 43.6]
nyc_temp_2012 = [37.3, 40.9, 50.9, 54.8, 65.1, 71.0, 78.8, 76.7, 68.8,
58.0, 43.9, 41.5]
months = range(1, 13)
plot(months, nyc_temp_2000, months, nyc_temp_2006, months, nyc_temp_2012)
show()

#P41: Adding a legend to the above graph
from pylab import legend
plot(months, nyc_temp_2000, months, nyc_temp_2006, months, nyc_temp_2012)
legend([2000, 2006, 2012])
show()

#P42: Adding a title, legend and labels to a graph
from pylab import plot, show, title, xlabel, ylabel, legend
plot(months, nyc_temp_2000, months, nyc_temp_2006, months, nyc_temp_2012)
title('Average monthly temperature in NYC')
xlabel('Month')
ylabel('Temperature')
legend([2000, 2006, 2012])
show()

#P43: Customizing the axes limits
nyc_temp = [53.9, 56.3, 56.4, 53.4, 54.5, 55.8, 56.8, 55.0, 55.3, 54.0,
56.7, 56.4, 57.3]
plot(nyc_temp, marker='o')
axis(ymin=0)
show()

#P43: Simple plot using pyplot
'''
Simple plot using pyplot
'''
import matplotlib.pyplot

def create_graph():
    x_numbers=[1,2,3]
    y_numbers=[2,4,6]
    matplotlib.pyplot.plot(x_numbers, y_numbers)
    matplotlib.pyplot.show()
    
if __name__ == '__main__':
    create_graph()

#P44: Importing matplotlib.pyplot as plt
'''
Simple plot using pyplot
'''
import matplotlib.pyplot as plt

def create_graph():
    x_numbers = [1, 2, 3]
    y_numbers = [2, 4, 6]
    plt.plot(x_numbers, y_numbers)
    plt.show()

if __name__ == '__main__':
    create_graph()

#P45: Saving the plots
from pylab import plot, savefig
x = [1, 2, 3]
y = [2, 4, 6]
plot(x, y)
savefig('mygraph.png')

#P46: Relationship between gravitational force and distance
'''
The relationship between gravitational force and
distance between two bodies
'''
import matplotlib.pyplot as plt
# draw the graph
def draw_graph(x, y):
    plt.plot(x, y, marker='o')
    plt.xlabel('Distance in meters')
    plt.ylabel('Gravitational force in newtons')
    plt.title('Gravitational force and distance')
    plt.show()
def generate_F_r():
    # generate values for r
    r = range(100, 1001, 50)
    # empty list to store the calculated values of F
    F = []
    # constant, G
    G = 6.674*(10**-11)
    # two masses
    m1 = 0.5
    m2 = 1.5
    # calculate Force and add it to the list, F
    for dist in r:
        force = G*(m1*m2)/(dist**2)
        F.append(force)
    # call the draw_graph function
    draw_graph(r, F)

if __name__=='__main__':
    generate_F_r()

#P50: Draw the trajectory of a projectile motion

'''
Draw the trajectory of a body in projectile motion
'''
from matplotlib import pyplot as plt
import math
def draw_graph(x, y):
    plt.plot(x, y)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Projectile motion of a ball')
    
def frange(start, final, interval):
    numbers = []
    while start < final:
        numbers.append(start)
        start = start + interval
    return numbers

def draw_trajectory(u, theta):
    theta = math.radians(theta)
    g = 9.8
    # Time of flight
    t_flight = 2*u*math.sin(theta)/g
    # find time intervals
    intervals = frange(0, t_flight, 0.001)
    # list of x and y coordinates
    x = []
    y = []
    for t in intervals:
        x.append(u*math.cos(theta)*t)
        y.append(u*math.sin(theta)*t - 0.5*g*t*t)
    draw_graph(x, y)
    
if __name__ == '__main__':
    try:
        u = float(input('Enter the initial velocity (m/s): '))
        theta = float(input('Enter the angle of projection (degrees): '))
    except ValueError:
        print('You entered an invalid input')
    else:        
        draw_trajectory(u, theta)
        plt.show()

#P53: Comparing the trajectory at different intial velocities
if __name__ == '__main__':
    # list of three different initial velocity
    u_list = [20, 40, 60]
    theta = 45
    for u in u_list:
        draw_trajectory(u, theta)
    # Add a legend and show the graph
    plt.legend(['20', '40', '60'])
    plt.show()

#P55: Quadratic function calculator
'''
Quadratic function calculator
'''
# assume values of x
x_values = [-1, 1, 2, 3, 4, 5]
for x in x_values:
    # calculate the value of the quadratic
    # function
    y = x**2 + 2*x + 1
    print('x={0} y={1}'.format(x, y))

## P58: Horizontal bar chart example

'''
Example of drawing a horizontal bar chart
'''

import matplotlib.pyplot as plt
def create_bar_chart(data, labels):
    # number of bars
    num_bars = len(data)
    # this list is the point on the y-axis where each
    # bar is centered. Here it will be [1, 2, 3..]
    positions = range(1, num_bars+1)
    plt.barh(positions, data, align='center')
    # set the label of each bar
    plt.yticks(positions, labels)
    plt.xlabel('Steps')
    plt.ylabel('Day')
    plt.title('Number of steps walked')
    # Turns on the grid which may assist in visual estimation
    plt.grid()
    plt.show()
    

if __name__ == '__main__':
    # Number of steps I walked during the past week
    steps = [6534, 7000, 8900, 10786, 3467, 11045, 5095]
    # Corresponding days
    labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    create_bar_chart(steps, labels)
    
   
    

##P50: Draw the trajectory of a projectile motion using list comprehensions (Appendix B)

'''
Draw the trajectory of a body in projectile motion
'''
from matplotlib import pyplot as plt
import math
def draw_graph(x, y):
    plt.plot(x, y)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Projectile motion of a ball')
    
def frange(start, final, interval):
    numbers = []
    while start < final:
        numbers.append(start)
        start = start + interval
    return numbers

def draw_trajectory(u, theta):
    theta = math.radians(theta)
    g = 9.8
    # Time of flight
    t_flight = 2*u*math.sin(theta)/g
    # find time intervals
    intervals = frange(0, t_flight, 0.001)
    # list of x and y coordinates
    x = [ u*math.cos(theta)*t for t in intervals]
    y = [u*math.sin(theta)*t - 0.5*g*t*t for t in intervals]
    draw_graph(x, y)
    
if __name__ == '__main__':
    try:
        u = float(input('Enter the initial velocity (m/s): '))
        theta = float(input('Enter the angle of projection (degrees): '))
    except ValueError:
        print('You entered an invalid input')
    else:        
        draw_trajectory(u, theta)
        plt.show()

