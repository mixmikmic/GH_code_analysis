from __future__ import division, print_function
#from math import *
import vpython as vp

#import ivisual as vp

import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

from itertools import product

# Simulation speed
RATE = 150
dt = 1./RATE

showFilteringDensity = True

# Set the scene

# Floor
W1 = 7
W2 = 7

floorHeight = -1.5

floor = vp.box(pos=vp.vector(0,floorHeight,0), length=W1, height=0.05, width=W2, color=vp.vector(0.4,0.4,0.4))

# Tiles
K = 36
radius = 0.15

PointList = [(2*(np.cos(th)+np.sin(th)), 0, 2*(np.cos(th)-np.sin(th)) ) for th in np.linspace(0,2*np.pi,K)]
L = len(PointList)

cols = [vp.color.white, vp.color.blue]
Cols = []
Estimate = []
for i in range(L):
    sx, sy, sz = PointList[i]
    s = vp.vector(sx,sy,sz)
    c = np.random.choice(range(len(cols)))
    Cols.append(c)
    vp.cylinder(pos=s, axis=vp.vector(0,-radius,0), color=cols[c], radius=radius)
    
    if showFilteringDensity:
        s2 = vp.vector(sx,sy+floorHeight-0.1,sz)
        cyl = vp.cylinder(pos=s2, axis=vp.vector(0,0.4,0), radius=radius)
        Estimate.append(cyl)

# Probability of staying on the same tile
ep = 0.4
# Probability of making an arbitrary jump
kidnap = 0.01
# Probability of correct observation
a = 0.99

# Set up the transition matrix
idx = [i for i in range(1,L)]+[0]
I = np.diag(np.ones(L))
A = (1-kidnap)*(ep*I + (1-ep)*I[:,idx]) + kidnap*np.ones((L,L))/L
C = np.zeros((2,L))
pred = np.ones(L)/L

for i in range(L):
    C[0,i] = a*(1 - Cols[i]) + (1-a)*Cols[i]
    C[1,i] = a*Cols[i] + (1-a)*(1 - Cols[i])

# Number of particles
N = 1

Obs = [] 
for i in range(N):
    o = vp.sphere(pos=vp.vector((i-N/2)/2.,floorHeight,0), color=vp.color.black, radius=radius)
    Obs.append(o)

## Each particle may move under a different gravitational force
g_earth = 19.8
T_period = 0.5

# Cur[i] is the discrete state index of the i'th particle
Cur = []

# Cnt[i] is the number of ticks after a new movement has started of the i'th particle
Cnt = []
# Ball objects
B = []


MAX = 0.7

nf = mpl.colors.Normalize(vmin=0, vmax=MAX, clip=True)
cmap = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=nf)

lamp = vp.local_light(pos=vp.vector(5,-4,0), color=vp.color.white)

for i in range(N):  
    cur = np.random.choice(range(L))
    sx, sy, sz = PointList[cur]
    x0 = vp.vector(sx,sy,sz)
    v0 = vp.vector(0,0,0)
    ball = vp.sphere(pos=x0, radius=radius, color=vp.color.yellow,
                 make_trail=True, interval=1, retain=RATE*T_period)
    ball.vel = v0
    ball.g = g_earth/(i+1)  # *(i+0.5)
    ball.T_period = T_period #*(i+1)
    ball.retain = RATE*ball.T_period
    cnt = ball.T_period/dt

    B.append(ball)
    Cur.append(cur)
    Cnt.append(cnt)


while 1:
    vp.rate (RATE)

    for i in range(N):        
        B[i].pos = B[i].pos + B[i].vel*dt
        if Cnt[i]>= B[i].T_period/dt/2.:
            Obs[i].color = vp.color.black
        
        if Cnt[i]>= B[i].T_period/dt:
            o = Cols[Cur[i]]
            Obs[i].color = cols[o]
            
            ## Select the new target index
            u = np.random.rand()
            if u<1-ep:
                nex = (Cur[i]+1)%L
            elif u>0.99:
                nex = np.random.choice(range(L))
            else:
                nex = (Cur[i])%L
                #nex = np.random.choice(range(L))

            
            if i==0:
                pred = C[o,:]*pred
                pred = pred/np.sum(pred)
                
                if showFilteringDensity:
                    for k in range(L):
                        col = cmap.to_rgba(pred[k])
                        vcol = vp.vector(col[0], col[1], col[2])
                        Estimate[k].color = vcol
                        Estimate[k].axis=vp.vector(0,pred[k]+0.15,0)
                        
                pred = A.dot(pred)
                
            
            ## Plan the jump
            sx, sy, sz = PointList[Cur[i]]
            tx, ty, tz = PointList[nex]
            v_vert = B[i].g*B[i].T_period/2 + (ty-sy)/B[i].T_period
                
            B[i].vel = vp.vector((tx-sx)/B[i].T_period, v_vert,(tz-sz)/B[i].T_period)   
            B[i].pos = vp.vector(sx,sy,sz)
            #
            Cur[i] = nex
            Cnt[i] = 0
        else:
            B[i].vel.y = B[i].vel.y - B[i].g*dt
            Cnt[i] +=1

from __future__ import division, print_function
#from math import *
import vpython as vp
import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl

from vpython_utilities import make_grid2D

# Simulation speed
RATE = 300
dt = 1./RATE

W1 = 1
W2 = 6
step = 0.5

n2 = int(W2/step)+1
n1 = int(W1/step)+1
PointList, sub2ind, ind2sub, edges, A = make_grid2D(n2,n1)

Trans = A/np.sum(A, axis=0, keepdims=True)
Trans = Trans.dot(Trans).dot(Trans)
L = len(PointList)

Y = []
for i in range(L):
    p = PointList[i]
    x = p[0]*step-W2/2.
    z = p[2]*step-W1/2.
    E = 2+ np.cos(2*pi*z/3)+np.sin(2*pi*x/5)+np.random.randn()/10.
    y = 2*np.exp(-1.1*E)
    PointList[i] = (x, 0, z)
    Y.append(y)
    
MAX = 1
MIN = 0
nf = mpl.colors.Normalize(vmin=MIN, vmax=MAX, clip=True)
cmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r, norm=nf)

#floor = box(pos=vector(0,-0.04,0), length=W1, height=0.05, width=W2, color=color.black)

wd = 0.4
radius = 0.2
maxY = max(Y)

for i in range(L):
    sx, sy, sz = PointList[i]
    s = vp.vector(sx,-radius,sz)
    #vp.sphere(pos=s, color=vp.color.cyan, radius=0.1)
    #vcol = vp.vector(0.9,0.9,0.9)
    col = cmap.to_rgba(Y[i]/maxY)
    vcol = vp.vector(col[0], col[1], col[2])
    #vp.cylinder(pos=s, axis=vp.vector(0,-0.1,0), color=vcol, radius=radius*np.sqrt(nf(sy))) 
    
    wd = step*np.sqrt(Y[i]/maxY)
    vp.box(pos=s,length=wd, height=0.05, width=wd, color=vcol )
    #s = vp.vector(sx,(sy-radius)/2.,sz)
    #vp.box(pos=s,length=wd, height=sy-radius, width=wd, color=vcol )

Cur = []

# Cnt[i] is the number of ticks after a new movement has started of the i'th particle
Cnt = []

B = []


g_earth = 49.8
T_period = 0.25

N = 2

for i in range(N):  
    cur = np.random.choice(range(L))
    sx, sy, sz = PointList[cur]
    x0 = vp.vector(sx,sy,sz)
    v0 = vp.vector(0,0,0)
    ball = vp.sphere(pos=x0, radius=radius, color=vp.color.yellow,
                 make_trail=True, interval=1, retain=RATE*T_period)
    ball.vel = v0
    ball.g = g_earth
    ball.T_period = T_period 
    ball.retain = RATE*ball.T_period
    cnt = ball.T_period/dt
    B.append(ball)
    Cur.append(cur)
    Cnt.append(cnt)

lamp = vp.local_light(pos=vp.vector(0,-1,0), color=vp.color.yellow)

    
def selectNextState(cur):
    pr = Trans[:,cur]
    nex = np.random.choice(range(L), p=pr) 
    lw = np.log(Trans[cur, nex]) - np.log(Trans[nex, cur])
    return nex, lw

def planJump(ball, curPos, nexPos):
    sx, sy, sz = curPos
    tx, ty, tz = nexPos
    v_vert = ball.g*ball.T_period/2 + (ty-sy)/ball.T_period                
    vel = vp.vector((tx-sx)/ball.T_period, v_vert,(tz-sz)/ball.T_period)   
    pos = vp.vector(sx,sy,sz)  
    return pos, vel


# Particle index of the Chain
pP = 0
# Particle index of the proposal
pQ = 1
B[pQ].make_trail = False
B[pQ].color = vp.vector(0.6,0.6,0.6)
B[pQ].radius = radius/2

# Is proposal ball moving?
pQmove = True
log_q_ratio = 0

while 1:
    vp.rate (RATE)
    
    B[pQ].pos = B[pQ].pos + B[pQ].vel*dt
    B[pP].pos = B[pP].pos + B[pP].vel*dt
    if Cnt[pQ]>= B[pQ].T_period/dt:
        
        if pQmove:
            accept = np.log(np.random.rand()) < log_q_ratio + np.log(Y[Cur[pQ]]) - np.log(Y[Cur[pP]])

            if accept:
                
                # pP jumps to new location
                B[pP].g = g_earth
                pos, vel = planJump(B[pP], PointList[Cur[pP]], PointList[Cur[pQ]])              
                B[pP].vel = vel 
                B[pP].pos = pos
                
                # pQ stays put
                B[pQ].vel = vp.vector(0,0,0)
                B[pQ].pos.x, B[pQ].pos.y, B[pQ].pos.z = PointList[Cur[pQ]]
                B[pQ].g  = 0

                Cur[pP] = Cur[pQ]
            else:
                # pP jumps vertically
                B[pP].g = g_earth
                pos, vel = planJump(B[pP], PointList[Cur[pP]], PointList[Cur[pP]])  
                B[pP].vel = vel
                B[pP].pos = pos


                # pQ disappears
                B[pQ].visible = False
                B[pQ].g = g_earth/10.
                pos, vel = planJump(B[pQ], PointList[Cur[pQ]], PointList[Cur[pP]])              
                B[pQ].vel = vel 
                B[pQ].pos = pos

                Cur[pQ] = Cur[pP]               
            pQmove = False
        else:
            B[pQ].visible = True

            nex, log_q_ratio = selectNextState(Cur[pP])
            # pP stays put
            B[pP].vel = vp.vector(0,0,0)
            B[pP].pos.x, B[pP].pos.y, B[pP].pos.z = PointList[Cur[pP]]
            B[pP].g  = 0

            # pQ jumps to new location
            B[pQ].g = g_earth/10.
            pos, vel = planJump(B[pQ], PointList[Cur[pP]], PointList[nex])              
            B[pQ].vel = vel 
            B[pQ].pos = pos

            Cur[pQ] = nex               
            
            pQmove = True

                
        Cnt[pP] = 0
        Cnt[pQ] = 0
                
    else:
        B[pP].vel.y = B[pP].vel.y - B[pP].g*dt
        B[pQ].vel.y = B[pQ].vel.y - B[pQ].g*dt
        Cnt[pP] +=1
        Cnt[pQ] +=1
         

