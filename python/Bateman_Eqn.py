#NAME: Bateman Equation
#DESCRIPTION: Radioactive decay chains and the Bateman equation.

import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib notebook')

from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider, fixed

def decay_cycle(N_t,N,N_d,P,data):
	for i in range(1,N_t+1):
		rand = np.random.rand(N)
		for j in range(N):
			for k in range(N_d):
				if rand[j] < P[k]:
					if data[j,i] == k:                    
						data[j,i:] = data[j,i:] + 1.0
						break
	return data

def counter(N_t,N_d,data):
	# Counted nucleii
	count = np.zeros((N_d,N_t+1))
	for i in range(N_d):
		count[i,:] = np.sum(data.astype('int64') == i,axis = 0)
	return count

global N,N_t,P

# Number of nucleii
N = 10000

# Number of daughters in chain
N_d = 4

# Probability of each decay
P = np.zeros((N_d))
P[0] = 0.25
P[1] = 0.025
P[2] = 0.075
P[3] = 0.001

# Number of time steps
N_t = 25

# Initialise plot
line_array = []
fig = plt.figure(figsize = (12,6))
ax = plt.subplot(111)
for i in range(N_d):
    line, = ax.plot([],[])
    line_array.append(line)
line, = ax.plot([],[],'-.')
line_array.append(line)
ax.set_ylabel('No. of Nuclei')
ax.set_xlabel('Time')

def update(event):
    # Data record array
    initial_state = np.zeros((N,N_t+1))

    data = decay_cycle(N_t,N,N_d,P,initial_state)

    count = counter(N_t,N_d,data)

    time = np.arange(0,N_t+1)
    stable = np.sum(count, axis = 0)
    for i in range(N_d):
        line_array[i].set_data(time,count[i,:])
    line_array[N_d].set_data(time,N-stable)
    ax.set_xlim([0,time[N_t]])
    ax.set_ylim([0,N])

def N_update(val):
    global N
    N = val
    display(fig)

def N_t_update(val):
    global N_t
    N_t = val
    display(fig)

def prob_update(val,i):
    global P
    P[i] = val
    display(fig)

widgets.interact(N_update,val = IntSlider(min=10, max=100000, step=1, value=N, description='Number of Nuclei'))
widgets.interact(N_t_update,val = IntSlider(min=1, max=250, step=1, value=N_t, description='Number of Time Steps'))

def prob_widget(val,name,rank):
    widgets.interact(prob_update,val = FloatSlider(min=0., max=1., step=0.001, value=val, description=name), i=fixed(rank))

prob_widg_list = ['Primary Decay Prob','Secondary Decay Prob','Tertiary Decay Prob','Quaternary Decay Prob']

for i in range(N_d):
    prob_widget(P[i],prob_widg_list[i],i)

run_button = widgets.Button(description="Run")
display(run_button)
run_button.on_click(update)

