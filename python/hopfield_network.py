from IPython.core.display import HTML

get_ipython().magic('matplotlib inline')
import nengo
from nengo.dists import Uniform
from nengo.utils.ensemble import response_curves
import matplotlib.pyplot as plt
import numpy as np

   
mode = 2
if mode == 1:
    N = 4      #number of neurons
    start =   np.random.rand(N)  
    #start =  np.array([0,1,0,1])   #single attractor
    #start =  np.array([0,1,0,0])   #All these vectors are just 1 bit flip away 
    #start =  np.array([1,1,0,1])   #from the single attractor in the network, so  
    #start =  np.array([0,1,1,1])   #using any of them as initial input will make  
    #start =  np.array([0,0,0,1])   #the network settle at the attractor 0101.
    weights = np.matrix('0 -1 1 -1; -1 0 -1 1; 1 -1 0 -1; -1 1 -1 0')
    
elif mode == 2:
    N = 5       #number of neurons
    start =   np.random.rand(N)
    #start =  np.array([0,1,1,0,1])   #first attractor
    #start =  np.array([0,1,1,0,0])   #Using any of these three vectors as initial input
    #start =  np.array([0,1,1,1,1])   #will make the network settle at the first attractor 01101
    #start =  np.array([0,1,0,0,1])   #since they are all just 1 bit flip away from it.
    
    #start =  np.array([1,0,1,0,1])   #second attractor
    #start =  np.array([1,0,1,0,0])   #Using any of these three vectors as initial input
    #start =  np.array([1,0,1,1,1])   #will make the network settle at the second attractor 10101
    #start =  np.array([1,0,0,0,1])   #since they are all just 1 bit flip away from it.
    
    weights = np.matrix('0 -2 0 0 0; -2 0 0 0 0; 0 0 0 -2 2; 0 0 -2 0 -2; 0 0 2 -2 0')

model = nengo.Network('Hopfield net')

with model:
    stim = nengo.Node(lambda t: start if (t<=.1) else [0]*N)
    ens = nengo.Ensemble(N, dimensions=N, 
                         encoders=np.eye(N), 
                         max_rates=Uniform(.999,.999),
                         intercepts=Uniform(-1,-1),
                         neuron_type=nengo.neurons.Sigmoid(tau_ref=1))
    
    conn = nengo.Connection(ens, ens, transform=weights, synapse=0)
    nengo.Connection(stim, ens, synapse=0)
    
    stim_p = nengo.Probe(stim)
    ens_p = nengo.Probe(ens.neurons, 'rates')   
    
sim = nengo.Simulator(model)
sim.run(3)
t = sim.trange()   

fig = plt.figure(figsize=(12, 3))
p0 = plt.subplot(1, 2, 1)
p0.plot(*response_curves(ens, sim));
p0.set_title("Response curves")

p1 = plt.subplot(1, 2, 2)
p1.matshow(weights);
p1.set_title("Weigths")

print "\nMode:", mode

print "\nWeights"
print weights

print "\nStimulus provided to the network (stim)"
print sim.data[stim_p][0]

print "\nFinal Ensemble Value (ens)"
print sim.data[ens_p][-1]

sim = nengo.Simulator(model)
sim.run(3)
t = sim.trange()

fig = plt.figure(figsize=(12, 3))
p0 = plt.subplot(1, 2, 1)
p0.plot(*response_curves(ens, sim));
p0.set_title("Response curves")

p1 = plt.subplot(1, 2, 2)
p1.matshow(weights);
p1.set_title("Weigths")

print "\nMode:", mode

print "\nWeights"
print weights

print "\nStimulus provided to the network (stim)"
print sim.data[stim_p][0]

print "\nFinal Ensemble Value (ens)"
print sim.data[ens_p][-1]

