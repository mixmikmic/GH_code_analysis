get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

import simpy
import itertools
import random
import pandas as pd
import numpy as np
from scipy.integrate import odeint

def dx(x,t):
    dx = 0.5*(u-x)
    return dx
t = 0
dt = 2
u = 12
odeint(dx,0,[t,t+dt])[-1][0]

class constant(object):
    def __init__(self,env,dt,yout):
        self.env = env
        self.dt = dt
        self.yout = yout
        self.action = env.process(self.run())
        
    def run(self):
        while True:
            yield self.env.timeout(self.dt)
            print(self.yout)
        
env = simpy.Environment()


con = constant(env,.1,2.3)
env.run(until=1)

con

u = 1.0
y = 0.0

class firstorder(object):
    
    def __init__(self,env,a,dt):
        self.env = env
        self.a = a
        self.dt = dt
        self.env.process(self.update)
        
    def update(self):
        while True:
            t = env.now
            print(t)
            yield env.timeout(dt)
        
        

env = simpy.Environment()
f = firstorder(env,2,.1)
f.run(until=3)
        



