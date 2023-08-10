def line_of_sight(radius, height):
    '''Returns the line-of-sight (LOS) horizon distance when given the radius of the planet
    in meters and the height about the surface in meters.'''
    d = (height**2 + (2 * radius * height))**0.5
    return d

import math

def distance_along_the_planet(radius, height):
    '''Returns the distance along the planet to the tangent point when given the radius and 
    height of the planet in meters.'''
    L = radius * math.acos(radius / (radius + height))
    return L

# Horizon distance in kilometers (Earth)
line_of_sight(6378000, 2) / 1000.0

# Horizon distance in kilometers (The Moon)
line_of_sight(1738000, 2) / 1000.0

# Line-of-sight reception distance in kilometers (Moon)
distance_along_the_planet(1738000, 50) / 1000.0

def roc_lunar(radius, height):
    '''Rate of change of the lunar radius for each additional meter of antenna height,
    given radius and height in meters.'''
    dddh = 0.5 * (100 + (2 * radius)) * (2500 + (2 * height * radius)) ** -0.5
    return dddh

roc_lunar(1738000, 50)

def roc_lunar_dist(radius, height):
    '''Rate of change of the distance to the lunar tower at the LOS position, 
    in meters, given radius and height in meters.'''
    dldh = radius / (2*radius*height)**0.5
    return dldh

roc_lunar_dist(1738000, 50)

