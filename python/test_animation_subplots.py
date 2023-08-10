#Standard libraries
import numpy as np
import matplotlib.pyplot as plt

plt.subplots(2,2)

#Setting up graph

fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2)
axis = [ax1,ax2,ax3,ax4]
for ax in axis:
    ax.set_ylim(-1,1)
    ax.set_xlim(-np.pi,np.pi)

#Setting up sine-wave

#This initializes the line object to x= [],y=[]

#IMPORT DISTINCTION IN PYTHON
#Try it fuure me if you don't believe me.
#line, --- means type(line) == matplolib
#line --- on the otherhand means type(line) == list

line, = ax1.plot([],[],lw = 1,c= 'k')
line2, = ax2.plot([],[],lw = 1,c= 'k')
line3, = ax3.plot([],[],lw = 1,c= 'k')

#the variable i denotes the value changing each frame of the animation

#Initializes the animation
def initialize_animation():
    line.set_data([],[])
    line2.set_data([],[])
    line3.set_data([],[])
    #line, returns a tuple, NOT matplotlib object
    return line,


k,dk,w,dw =  -3,0.3,-1,0.01


#Number of linespace spacings
length = 250
x = np.linspace(-np.pi,np.pi,length)

#The frame generation function
def animate(t):    
    global length,x
    global k,dk,w,dw
    #omega is used for omega*t distance offset
    y1 = np.sin(2*np.pi*((k+dk)*x + (w + dw)*t))
    y2 = np.sin(2*np.pi*((k-dk)*x + (w - dw)*t))
    y = y1+y2
    
    
    line.set_data(x,y)
    line2.set_data(x,y1)
    line3.set_data(x,y2)
    
    return line, line2,line3,

#Animation plotting libraries

from IPython.display import HTML
import matplotlib.animation as animation

#Genrating the animation

ani = animation.FuncAnimation(fig, animate, np.arange(1,len(x)), interval=30, blit=True, init_func=initialize_animation)

#Generates HTML plot
HTML(ani.to_html5_video())


#To save animation, comment out the sys.exit()
#import sys
#sys.exit()
    
#Save animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps = 30,extra_args=['-vcodec', 'libx264'])
dpi = 100

ani.save('beats_animation.mp4', writer = writer,dpi = dpi)

