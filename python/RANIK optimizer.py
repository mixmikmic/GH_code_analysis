import numpy as np
import tensorflow as tf # only these two packages required
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(None))
y = tf.square(x)

y

sess = tf.InteractiveSession()

y.eval(feed_dict={x: [1, 2, 3, 4, 5]})

# define the gradients of y with respect to x
dy_dx = tf.squeeze(tf.stack(tf.gradients(y, x)))

dy_dx

dy_dx.eval(feed_dict={x: 4})

dy_dx.eval(feed_dict={x: [1, 2, 3, 4, 5]})

g = 9.8; mu = 0.0

acceleration = (g * (dy_dx)) / tf.sqrt(1 + tf.square(dy_dx))

acceleration.eval(feed_dict={x: 100})

start_position = 300

position = start_position; velocity = 0 # initial velocity is 0
x_values, y_values = [position], []

for _ in range(600):
    print "X = " + str(position) + " V = " + str(velocity)
    y_value, a = sess.run((y, acceleration), feed_dict={x: position})
    print "obtained_acceleration = " + str(a) + "\n"
    
    # now update the velocity and the position
    velocity = velocity + a
    position = position - (velocity + (a / 2))
    
    x_values.append(position); y_values.append(y_value)

plt.figure().suptitle("Position values:")
plt.plot(x_values);

position

plt.figure().suptitle("Y-function values:")
plt.plot(y_values);

position = start_position; velocity = 0 # initial velocity is 0
x_values, y_values = [position], []

for _ in range(600):
    print "X = " + str(position) + " V = " + str(velocity)
    slope, y_value, a = sess.run((dy_dx, y, acceleration), feed_dict={x: position})
    print "obtained_acceleration = " + str(a)
    
    # define dt (interval of time as a function of the slope:)
    dt = min(abs(slope), 1)
    if(dt != 1):
        print "Current dt = " + str(dt) + " Current slope = " + str(slope) + "\n"
    
    # now update the velocity and the position
    velocity = velocity + (a * dt)
    position = position - ((velocity * dt) + ((a / 2) * (dt ** 2)))
    
    x_values.append(position); y_values.append(y_value)

plt.figure().suptitle("Position values:")
plt.plot(x_values);

position

position = start_position; velocity = 0 # initial velocity is 0
x_values, y_values = [position], []

for _ in range(1000):
    print "X = " + str(position) + " V = " + str(velocity)
    slope, y_value, a = sess.run((dy_dx, y, acceleration), feed_dict={x: position})
    print "obtained_acceleration = " + str(a)
    
    # define dt (interval of time as a function of the slope:)
    dt = abs(slope)
    if(dt != 1):
        print "Current dt = " + str(dt) + " Current slope = " + str(slope) + "\n"
    
    # now update the velocity and the position
    velocity = velocity + (a * dt)
    position = position - ((velocity * dt) + ((a / 2) * (dt ** 2)))
    
    x_values.append(position); y_values.append(y_value)

plt.figure().suptitle("Position values:")
plt.plot(x_values);

position = start_position; velocity = 0 # initial velocity is 0
x_values, y_values = [position], []

for _ in range(300):
    print "X = " + str(position) + " V = " + str(velocity)
    slope, y_value, a = sess.run((dy_dx, y, acceleration), feed_dict={x: position})
    print "obtained_acceleration = " + str(a)
    
    # define dt (interval of time as a function of the slope:)
    dt = np.log(abs(slope))
    if(dt != 1):
        print "Current dt = " + str(dt) + " Current slope = " + str(slope) + "\n"
    
    # now update the velocity and the position
    velocity = velocity + (a * dt)
    position = position - ((velocity * dt) + ((a / 2) * (dt ** 2)))
    
    x_values.append(position); y_values.append(y_value)

plt.figure().suptitle("Position values:")
plt.plot(x_values);

def synthetic_function(x, a):
    '''
        calculates a synthetic function of the given x. for a value of a
    '''
    x, a = float(x), float(a) # just to make sure that we always get good precision
    y = np.sqrt(1 - ((a ** 2) / ((np.abs(x) + a) ** 2)))
        
    return y

plot_values = [synthetic_function(val, np.log(abs(val) + 1)) for val in range(100)]
plt.plot(plot_values);

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

start_position = 1000

position = start_position; velocity = 0 # initial velocity is 0
x_values, y_values = [position], []

for cur_step in range(400):
    print "current_step: " + str(cur_step)
    print "X = " + str(position) + " V = " + str(velocity)
    slope, y_value, a = sess.run((dy_dx, y, acceleration), feed_dict={x: position})
    print "obtained_acceleration = " + str(a)
    
    # define dt (interval of time as a function of the slope:)
    dt = min(np.log(np.abs(slope) + 1), 0.8)
    if(dt != 1):
        print "Current dt = " + str(dt) + " Current slope = " + str(slope) + "\n"
    
    # now update the velocity and the position
    velocity = velocity + (a * dt)
    position = position - ((velocity * dt) + ((a / 2) * (dt ** 2)))
    
    x_values.append(position); y_values.append(y_value)

plt.figure().suptitle("Position values:")
plt.plot(x_values);

plt.figure().suptitle("Cost values:")
plt.plot(y_values);

print position

sess.close() # close the earlier session

tf.reset_default_graph() # reset the earlier graph

# define the placeholder for the input:
x = tf.placeholder(tf.float64, shape=(None))

# define the y function
y = tf.square(x)

# define the dy_dx here:
dy_dx = tf.squeeze(tf.stack(tf.gradients(y, x))) # gradients returns a list, so first stack it and then squeeze it

g = 9.8

# define the acceleration here:
acceleration = -(2 * g * (dy_dx)) / (1 + tf.square(dy_dx)) # this is the changed accelaration
# this term corresponds to: accn = g * cos(x) * sin(x) 
# This accounts for the slope and it's fundamental nature of pulling the body towards the center

sess = tf.InteractiveSession() # create a fresh new session

# check if y is giving correct values:
print y.eval(feed_dict={x: 3}) # should return 256

# check if dy_dx is giving correct values:
print dy_dx.eval(feed_dict={x: 3}) # should return 32 (since slope = 2x)

# check if the acceleration is giving consisten values:
print acceleration.eval(feed_dict={x: 0.4}) # Ok, I have checked on calculator and this does make sense

# temporarily change 
start_position = 0.02; g = 9.8; epsilon = 1

position = start_position; velocity = 0 # initial velocity is 0
x_values, y_values = [position], []

limit = 500

cur_step = 0; done = False
while(not done):
    print "current_step: " + str(cur_step)
    print "X = " + str(position) + " V = " + str(velocity)
    slope, prev_y_value, a = sess.run((dy_dx, y, acceleration), feed_dict={x: position})
    print "obtained_acceleration = " + str(a)
    print "Current slope = " + str(slope) 
    
    # define dt (interval of time as a function of the slope:)
    dt = 1 # (np.square(slope) / (1 + np.square(slope)))
    # time interval is linearly proportional to the slope of the land
    
    print "Current dt = " + str(dt) + "\n"
    
    # calculate the old (main_projected) velocity
    old_velocity = velocity * np.sqrt(1 + np.square(slope))
    
    # now update the velocity and the position
    position = position + ((velocity * dt) + ((a / 2) * (dt ** 2)))
    new_slope, new_y_value = sess.run((dy_dx, y), feed_dict={x: position})
    
    velocity = (np.sign(velocity + (a * dt)) * 
                    (np.sqrt(max((np.square(old_velocity) + (2 * g * (prev_y_value - new_y_value))), 0)))) / np.sqrt(1 + np.square(new_slope))
    
    x_values.append(position); y_values.append(y_value)
    
    if(dt == 0 or cur_step == limit):
        done = True
    
    # increment the current_step
    cur_step += 1

plt.figure().suptitle("Position values:")
plt.plot(x_values);

plt.figure().suptitle("Cost_value: ")
plt.plot(y_values);

y_values[-1]

position

