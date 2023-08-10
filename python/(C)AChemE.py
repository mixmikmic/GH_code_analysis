get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import transforms
import matplotlib.font_manager as fm

from IPython.display import Image, display
Image(filename='c-logo-polar-background.png')

n_circles = 10

distance_from_center = 1.5
size_circle = 600

start_angle = np.deg2rad(115) # [deg]->[rad]
final_angle = np.deg2rad(305) # [deg]->[rad]

hex_color = "#467B99"

r = distance_from_center * np.ones(n_circles) #vectorized set of radius distance from center
theta =  np.linspace(start_angle,final_angle,n_circles) # vector with angles
area = size_circle * np.pi * np.ones(n_circles) # vetctor with size of areas


fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0, 0, 1, 1], polar=True)


ax.scatter(theta, r, color=hex_color, s=area)  # plot circles

# Add the letter C as text (this is a little bit quircky)
ax.text(0.7, 0.445, 'C', color='#467B99', fontsize=350, family='Century Gothic',
               ha='right', va='center', transform=ax.transAxes, weight='heavy')

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_axes([0, 0, 1, 1], polar=True)

def anular_disks(n_circles, distance_from_center,size_circle, start_angle, final_angle, hex_color):
    """ Creates an array of circles"""
    
    r = distance_from_center * np.ones(n_circles) #vectorized set of radius 
    theta =  np.linspace(start_angle,final_angle,n_circles) # vector with angles
    area = size_circle * np.pi * np.ones(n_circles) # vetctor with size of areas

    plt.scatter(theta, r, color=hex_color, s=area)  # plot circles
    
    ax.text(0.66, 0.457, 'C', color='#467B99', fontsize=300, family='Century Gothic',
               ha='right', va='center', transform=ax.transAxes, weight='bold')
    

# anular_disks(n_circles, distance_from_center,size_circle, start_angle, final_angle, hex_color)

fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0, 0, 1, 1], polar=True)

n_circles = 10

start_angle_circles = [113, 105, 97]
final_angle_circles = [305, 295, 287]

distance_from_center_circles = [1.6, 1.9, 2]
sizes_diff_circles = [400, 200, 70]

colors_circles = ["#467B99", "#6BA0C2", "#739FF6"]

for i in range(0,3):
    distance_from_center = distance_from_center_circles[i]
    size_circle = sizes_diff_circles[i]

    start_angle = np.deg2rad(start_angle_circles[i]) # [deg]->[rad]
    final_angle = np.deg2rad(final_angle_circles[i]) # [deg]->[rad]

    hex_color = colors_circles[i]

    anular_disks(n_circles, distance_from_center,size_circle, start_angle, final_angle, hex_color)


plt.axis("off") # removes the axis (comment line in order to see what's going on)

#Set the size of our figure
axalpha = 0.05
figcolor = 'white'
dpi = 80
fig = plt.figure(figsize=(5, 5),dpi=dpi)
fig.figurePatch.set_edgecolor(figcolor)
fig.figurePatch.set_facecolor(figcolor)

# Define the axis of our figure
def add_background():
    ax = fig.add_axes([0., 0., 1., 1.])
    
    #And hide them
    ax.set_axis_off()
    return ax

# Let's get our Logo
    
def anular_disks(n_circles, distance_from_center,size_circle, start_angle, final_angle, hex_color):
    """ Creates an array of circles"""
    
    r = distance_from_center * np.ones(n_circles) #vectorized set of radius 
    theta =  np.linspace(start_angle,final_angle,n_circles) # vector with angles
    area = size_circle * np.pi * np.ones(n_circles) # vetctor with size of areas

    plt.scatter(theta, r, color=hex_color, s=area)  # plot circles

    ax.text(2.31, 0.47, 'CA           E', color='#467B99', fontsize=140, family='Century Gothic',
               ha='right', va='center', alpha=4.0, transform=ax.transAxes, weight='heavy')
    ax.text(1.21, 0.47, '  C', color='#6B727A', fontsize=140, family='Century Gothic',
               ha='right', va='center', alpha=1.0, transform=ax.transAxes, weight='heavy')
    ax.text(2.085, 0.47, '  hem', color='#6B727A', fontsize=140, family='Century Gothic',
               ha='right', va='center', alpha=1.0, transform=ax.transAxes)

    

#anular_disks(n_circles, distance_from_center,size_circle, start_angle, final_angle, hex_color)

fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0., 0., 1., 1.], polar=True)


n_circles = 10

start_angle_circles = [113, 105, 97]
final_angle_circles = [305, 295, 287]

distance_from_center_circles = [1.5*2, 1.95*2, 2.3*2]
sizes_diff_circles = [300/2, 200/2, 70/2]

colors_circles = ["#467B99", "#6BA0C2", "#739FF6"]

for i in range(0,3):
    distance_from_center = distance_from_center_circles[i]
    size_circle = sizes_diff_circles[i]

    start_angle = np.deg2rad(start_angle_circles[i]) # [deg]->[rad]
    final_angle = np.deg2rad(final_angle_circles[i]) # [deg]->[rad]

    hex_color = colors_circles[i]

    anular_disks(n_circles, distance_from_center,size_circle, start_angle, final_angle, hex_color)

# Plotting the results
plt.axis("off") # removes the axis (try to comment this line in order to see what's going on)


