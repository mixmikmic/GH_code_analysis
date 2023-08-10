#from plotly.graph_objs import Scatter, Layout
import matplotlib.pyplot as plt
import plotly
import numpy as np
import random
import hrr
import math
from plotly.graph_objs import Scatter, Layout, Surface
plotly.offline.init_notebook_mode(connected=True)

def log_transform(error):
    return math.copysign(1.0,error)*math.log(math.fabs(error)+1,2)

def argmax(arr_3d,outer,inner):
    max_row = outer[0]
    max_col = inner[0]
    max_x = 0
    #max_value = arr_2d[0,0]
    max_value = arr_3d[outer[0],inner[0],0]
    for row in range(arr_3d.shape[0]):
        if row not in outer:
            continue
        for col in range(arr_3d.shape[1]):
            if col not in inner:
                continue
            for x in range(arr_3d.shape[2]):
                if arr_3d[row,col,x] > max_value:
                    max_value = arr_3d[row,col,x]
                    max_row,max_col,max_x = row,col,x
    return list((max_row,max_col,max_x))

def context_check(outer,inner):
    return outer==0 and inner==0

def TD(ntrials,lrate):
    n = 64000
    nactions = 2 # number of actions
    nwm_o = 2 # number of outer wm slots
    nwm_i = 2 # number of inner wm slots
    nsig_o = 2 # number of outer external signals
    nsig_i = 2 # number of inner external signals
    
    ## reward matrix, reward given at 0,0,0
    reward = np.zeros((nsig_o+1,nsig_i+1,nactions))
    reward[0,0,0] = 1
    
    ## hrr for actions and states
    actions = hrr.hrrs(n,nactions)
    
    ## identity vector
    hrr_i = np.zeros(n)
    hrr_i[0] = 1
    
    ## external outer
    sig_outer = hrr.hrrs(n,nsig_o)
    sig_outer = np.row_stack((sig_outer,hrr_i))
    
    ## external inner
    sig_inner = hrr.hrrs(n,nsig_i)
    sig_inner = np.row_stack((sig_inner,hrr_i))
    
    ## Outer WorkingMemory
    wm_outer = hrr.hrrs(n,nwm_o)
    wm_outer = np.row_stack((wm_outer,hrr_i))
    
    ## Inner WorkingMemory
    wm_inner = hrr.hrrs(n,nwm_i)
    wm_inner = np.row_stack((wm_inner,hrr_i))
    
    ## precompute action,wm_o,wm_i,sig_o,sig_i
    #sa = hrr.oconvolve(actions,states)
    wm = hrr.oconvolve(hrr.oconvolve(wm_inner,wm_outer),actions)
    external = hrr.oconvolve(sig_inner,sig_outer)
    s_a_c_c_wm_wm = hrr.oconvolve(wm,external)
    s_a_c_c_wm_wm = np.reshape(s_a_c_c_wm_wm,(nsig_o+1,nsig_i+1,nwm_o+1,nwm_i+1,nactions,n))
    
    ## weight vector and bias
    W = hrr.hrr(n)
    bias = 0
    
    ## eligibilty trace, epsilon value and number of steps
    #eligibility = np.zeros(n)
    epsilon = .01
    nsteps = 10
    
    for trial in range(ntrials):
        # cue signal
        if trial%10==0:
            outer_signal = random.randrange(0,nsig_o)
        # probe signal
        inner_signal = random.randrange(0,nsig_i)
        
        ## sets the context for later use
        outer = outer_signal
        inner = inner_signal
        
        values = np.dot(s_a_c_c_wm_wm[outer_signal,inner_signal,:,:,:,:],W) + bias
        possible_outer_wm = np.unique(np.array([2,outer_signal]))
        possible_inner_wm = np.unique(np.array([2,inner_signal]))
        wm_wm_action = argmax(values,possible_outer_wm,possible_inner_wm)
        current_outer_wm = wm_wm_action[0]
        current_inner_wm = wm_wm_action[1]
        action = wm_wm_action[2]
        
        ## epsilon soft policy
        if random.random() < epsilon:
            action = random.randrange(0,nactions)
            current_outer_wm = random.choice(possible_outer_wm)
            current_inner_wm = random.choice(possible_inner_wm)
        
        values = values[current_outer_wm,current_inner_wm,action]
        
        #####
        r = reward[outer,inner,action]
        error = r - values
        W += lrate*log_transform(error)*s_a_c_c_wm_wm[outer_signal,inner_signal,current_outer_wm,current_inner_wm,action,:]
        #print('outer:',outer,'inner:',inner,'r:',r)
        if trial%1000==0:
            V1 = list(map(lambda x: np.dot(x,W)+bias, s_a_c_c_wm_wm[0,0,0,0,:,:]))
            V2 = list(map(lambda x: np.dot(x,W)+bias, s_a_c_c_wm_wm[1,1,1,1,:,:]))
            V3 = list(map(lambda x: np.dot(x,W)+bias, s_a_c_c_wm_wm[1,1,0,0,:,:]))
            V4 = list(map(lambda x: np.dot(x,W)+bias, s_a_c_c_wm_wm[0,0,1,1,:,:]))
            V5 = list(map(lambda x: np.dot(x,W)+bias, s_a_c_c_wm_wm[0,1,0,1,:,:]))
            V6 = list(map(lambda x: np.dot(x,W)+bias, s_a_c_c_wm_wm[1,0,1,0,:,:]))
            
            plotly.offline.iplot([
            dict(x=[x for x in range(len(V1))] , y=V1, type='scatter',name='A,X and outerwm_A innerwm_X'),
            dict(x=[x for x in range(len(V1))] , y=V2, type='scatter',name='B,Y and outerwm_B innerwm_Y'),
            dict(x=[x for x in range(len(V1))] , y=V3, type='scatter',name='B,Y and outerwm_A innerwm_X'),
            dict(x=[x for x in range(len(V1))] , y=V4, type='scatter',name='A,X and outerwm_B innerwm_Y'),
            dict(x=[x for x in range(len(V1))] , y=V5, type='scatter',name='A,Y and outerwm_A innerwm_Y'),
            dict(x=[x for x in range(len(V1))] , y=V6, type='scatter',name='B,X and outerwm_B innerwm_X'),
            ])
            

TD(10000,.3)
## trials,lrate



