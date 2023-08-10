import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Parameters
dt = 1
dx = 0.01
#How many seconds to run the program for
t = 200*dt

#Boundary conditions
x0,T0 = 0,1
xn,Tn = 1,1

x = np.r_[np.arange(x0,xn,dx), [xn]]

#Inital condition
T = np.zeros(len(x))
T[0] = T0
T[-1] = Tn
#Numerical computation number
r = 0.01

plt.plot(T)

plt.grid('off')
plt.show()

len(T)

fig = plt.figure()
ax = plt.axes(xlim=(x0,xn),ylim = (min(T),max(T)))
line, = ax.plot([],[],lw = 1,c= 'k')



def animate(step):    

    for i in range(1,len(T)-1):
        #Ti
        Ti = np.copy(T)
        #Ti+1
        T[i] = r*(Ti[i+1]+Ti[i-1]) + (1-2*r)*Ti[i]
    
    line.set_data(x,T)
    return line,

#Animation plotting libraries

from IPython.display import HTML
import matplotlib.animation as animation

ani = animation.FuncAnimation(fig, animate, np.arange(0,t), interval=30, blit=True)

#Generates HTML plot
HTML(ani.to_html5_video())

