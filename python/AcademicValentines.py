# Credits: http://www.walkingrandomly.com/?p=5964 
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import seaborn as sns
from JSAnimation import IPython_display


fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), 
              ylim=(-2, 2))
line, = ax.plot([], [], 
                lw=2, 
                color='red')

time_text1 = ax.text(-0.5, 1.5, 
                     "I've read", 
                     fontsize=18, 
                     color='black')
time_text2 = ax.text(-0.5, -1.7, 
                     "your work!", 
                     fontsize=18, 
                     color='black')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = np.linspace(-2, 2, 1000)
    y = (np.sqrt(np.cos(x))*np.cos(i*x)         +np.sqrt(np.abs(x))-0.7)*(4-x*x)**0.01
    line.set_data(x, y)
    time_text1.set_position([-np.sin(i/8.0)-0.25,1.5])
    time_text2.set_position([-np.sin(i/8.0)-0.25,-1.7])
    return line,

animation.FuncAnimation(fig, animate, init_func=init,
                        frames=100, interval=30)

