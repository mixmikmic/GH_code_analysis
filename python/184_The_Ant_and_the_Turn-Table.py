def ant_radial_motion(time):
    '''Returns the radial motion of the ant in centimeters, with respect to time, 
    when given time in minutes.'''
    return 5 * time

# Centimeters traveled in 15 seconds.
ant_radial_motion(0.25)

# Centimeters traveled in 1 minute.
ant_radial_motion(1)

import math

def ant_travel_distance(time):
    '''Returns the length of the ant's arc, in centimeters, when given time in minutes.'''
    arc_length = 5 * ((math.sinh(2 * time) / 4) + (time / 2))
    return arc_length

ant_travel_distance(1)

