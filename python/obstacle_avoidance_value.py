from simulator_interface import open_session, close_session
simulator, epuck = open_session(n_epucks=1)

close_session(simulator, epuck)

from time import sleep

# First detach all behavior on both robots:
epuck.detach_all_behaviors()
epuck.detach_all_routines()
epuck.stop()
    
# define the obstacle_avoidance behavior with a weight of 1. This is indicated by the third value returned by the function:
def obstacle_avoidance(robot):
    left, right = robot.prox_activations(tracked_objects=["20cm", "Tree", "Cup"])
    robot.value = 1. - (left + right ) / 2.0
    left_wheel = 1 - right
    right_wheel = 1 - left
    return left_wheel, right_wheel

def image_value(robot): 
    img = robot.camera_image()
    epuck.add_log("value", epuck.value)
    epuck.add_log("image", img)


# Attach and start both behaviors on both robots:
epuck.max_speed = 10.
epuck.attach_behavior(obstacle_avoidance, freq=10.0)
epuck.attach_routine(image_value, freq = 1.0)
epuck.start_all_behaviors()
sleep(1)
epuck.start_all_routines()

epuck.max_speed = 20.

epuck.detach_all_behaviors()
epuck.detach_all_routines()
epuck.stop()

values = epuck.get_log("value")
images = epuck.get_log("image")

get_ipython().magic('pylab inline')

i = argmin(values)
print values[i]
imshow(images[i], origin='lower')

