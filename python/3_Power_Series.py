import matplotlib.pyplot as plt
H=1 # m
gamma= 0.2 # a 20 percent loss of bounce energy, per bounce
s = [2*H, 2*H*(1-gamma)]
plt.plot(s,'d',clip_on=False)
plt.xlabel('Bounce')
plt.ylabel('Bounce path length $s$ (m)')
plt.xticks([0,1])
plt.show()

import numpy as np

bounces = np.arange(10)
s=[]
for bounce in bounces:
    s.append(2*H*(1-gamma)**bounce)                        
S = np.cumsum(s)
plt.plot(S,'d',clip_on=False)
plt.xlabel('Bounce')
plt.ylabel('Total bounce path length $S$ (m)')
plt.show()

gamma= 0.01 
bounces = np.arange(10)
s=[] # s is the path length of a single bounce
for bounce in bounces:
    s.append(2*H*(1-gamma)**bounce)                        
S = np.cumsum(s) # the cumulative sum of the path lengths
plt.plot(S,'d',clip_on=False,label='bounces')
plt.plot(2*(bounces+1)*H,label='approx.')
plt.xlabel('bounce')
plt.ylabel('Total bounce path length $S$ (m)')
plt.legend(loc=4)
plt.show()

gamma= 0.5
bounces = np.arange(10)
s=[] # s is the path length of a single bounce
for bounce in bounces:
    s.append(2*H*(1-gamma)**bounce)                        
S = np.cumsum(s) # the cumulative sum of the path lengths
plt.plot(S,'d',clip_on=False,label='bounces')
plt.plot(np.ones(len(bounces))*2*H/gamma, label='approx.')
plt.ylim([1,5])
plt.xlabel('Bounce')
plt.ylabel('Total bounce path length $S$ (m)')
plt.legend(loc=4)
plt.show()

gamma= 0.3
N=10
g = 9.8 #m/s^2
H=1
bounces = np.arange(N)
t=[] # t is the travel time of a single bounce
for bounce in bounces:
    t.append(np.sqrt(8*H/g)*np.sqrt(1-gamma)**bounce)                        
T = np.cumsum(t) # the cumulative sum of the path lengths
plt.plot(T,'d',clip_on=False,label='bounces')
plt.xlabel('Bounce')
plt.ylabel('Total bounce travel time $T$ (s)')
plt.show()

gamma= 0.3
N=30
g = 9.8 #m/s^2
H=1
bounces = np.arange(N)
t=[] # t is the time for a single bounce
for bounce in bounces:
    t.append(np.sqrt(8*H/g)*np.sqrt(1-gamma)**bounce)                        
T = np.cumsum(t) # the cumulative sum of the bounce times
plt.plot(T,'d',clip_on=False,label='series')
plt.plot(np.ones(len(bounces))*np.sqrt(8*H/g)/(1-np.sqrt(1-gamma)),label='analytic')
plt.xlabel('Bounce')
plt.ylabel('Total bounce time $T$ (s)')
plt.legend(loc=4)
plt.show()

gamma= 0.1
N=100
g = 9.8 #m/s^2
H=1
bounces = np.arange(N)
t=[] # t is the time for a single bounce
for bounce in bounces:
    t.append(np.sqrt(8*H/g)*np.sqrt(1-gamma)**bounce)                        
T = np.cumsum(t) # the cumulative sum of the bounce times
plt.plot(T,'d',clip_on=False,label='series')
plt.plot(np.ones(len(bounces))*np.sqrt(8*H/g)/(1-np.sqrt(1-gamma)),label='analytic')
plt.plot(np.ones(len(bounces))*np.sqrt(8*H/g)*2/gamma,label='Taylor')
plt.xlabel('Bounce')
plt.ylabel('Total bounce time $T$ (s)')

plt.legend(loc=4)
plt.show()

