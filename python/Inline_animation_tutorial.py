#NAME: Inline Animation Tutorial
#DESCRIPTION: How to use Matplotlib Animations and PyCav.display for inline notebook animation

# Use these two lines on environments without displays i.e. servers
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as anim

import pycav.display as display

import numpy as np

x = np.linspace(0.,10.,101)
t = np.linspace(0.,10.,51)

sin_x = np.zeros((101,51))

for i in range(51):
    sin_x[:,i] = np.sin(np.pi*x-np.pi*t[i])

fig = plt.figure(figsize = (7,7))
ax = plt.subplot(111)
line = ax.plot(x,sin_x[:,0])[0]
    
def nextframe(arg):
    line.set_data(x,sin_x[:,arg+1])
    
animate1 = anim.FuncAnimation(fig,nextframe,interval = 100,frames = 50, repeat = False)

animate1 = display.create_animation(animate1,temp = True)

display.display_animation(animate1)

animate1 = display.create_animation(animate1,fname = 'example.mp4')
display.display_animation(animate1)

animate1 = display.create_animation(animate1,fname = 'example.mp4', overwrite = False)
display.display_animation('example.mp4')



