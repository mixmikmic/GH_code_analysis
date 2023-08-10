get_ipython().magic('matplotlib inline')
import numpy as np
import nengo
import pylab

import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.dists.Combined = nengo.utils.function_space.Combined
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

domain = np.linspace(-1, 1, 2000)

def gaussian(mag, mean, sd):
    return mag * np.exp(-(domain-mean)**2/(2*sd**2))

pylab.plot(domain, gaussian(mag=1, mean=0, sd=0.1))
pylab.show()

space = []
for i in range(100):
    space.append(gaussian(mag=np.random.uniform(0,1), 
                          mean=np.random.uniform(-1,1), 
                          sd=np.random.uniform(0.1, 0.3)))
pylab.plot(domain, np.array(space).T)
pylab.show()

space = nengo.dists.Function(gaussian,
                             mag=nengo.dists.Uniform(0, 1),
                             mean=nengo.dists.Uniform(-1, 1),
                             sd=nengo.dists.Uniform(0.1, 0.3))
data = space.sample(100)
pylab.plot(domain, data.T)
pylab.show()

model = nengo.Network()
with model:
    stim = nengo.Node(gaussian(mag=1, mean=0.5, sd=0.1))
    ens = nengo.Ensemble(n_neurons=200, dimensions=2000,
                         encoders=space,
                         eval_points=space,
                        )
    nengo.Connection(stim, ens)
    probe_func = nengo.Probe(ens, synapse=0.03)
sim = nengo.Simulator(model)
sim.run(0.2)
pylab.plot(domain, sim.data[probe_func][-1])
pylab.figure()
pylab.imshow(sim.data[probe_func], extent=(-1,1,0.2,0), aspect=10.0)
pylab.ylabel('time')
pylab.show()

model = nengo.Network()
with model:
    stim = nengo.Node(gaussian(mag=1, mean=0.5, sd=0.2))
    ens = nengo.Ensemble(n_neurons=100, dimensions=2000,
                         encoders=space,
                         eval_points=space)
    nengo.Connection(stim, ens)
        
    peak = nengo.Ensemble(n_neurons=50, dimensions=1)
    def find_peak(x):
        return domain[np.argmax(x)]
    nengo.Connection(ens, peak, function=find_peak)
    
    probe_peak = nengo.Probe(peak, synapse=0.03)
    
sim = nengo.Simulator(model)
sim.run(0.2)
pylab.plot(sim.trange(), sim.data[probe_peak])
pylab.show()

fs = nengo.FunctionSpace(
        nengo.dists.Function(gaussian,
                             mag=nengo.dists.Uniform(0, 1),
                             mean=nengo.dists.Uniform(-1, 1),
                             sd=nengo.dists.Uniform(0.1, 0.3)), 
        n_basis=20)

model = nengo.Network()
with model:
    stim = nengo.Node(fs.project(gaussian(mag=1, mean=0.5, sd=0.1)))
    ens = nengo.Ensemble(n_neurons=100, dimensions=fs.n_basis,
                         encoders=fs.project(space),
                         eval_points=fs.project(space))
    nengo.Connection(stim, ens)
    probe_func = nengo.Probe(ens, synapse=0.03)
sim = nengo.Simulator(model)
sim.run(0.2)
pylab.plot(domain, fs.reconstruct(sim.data[probe_func][-1]))
pylab.figure()
pylab.imshow(fs.reconstruct(sim.data[probe_func]), extent=(-1,1,0.2,0), aspect=10.0)
pylab.ylabel('time')
pylab.show()

model = nengo.Network()
with model:
    stim = nengo.Node(fs.project(gaussian(mag=1, mean=0.5, sd=0.2)))
    ens = nengo.Ensemble(n_neurons=100, dimensions=fs.n_basis,
                         encoders=fs.project(space),
                         eval_points=fs.project(space))
    nengo.Connection(stim, ens)
        
    peak = nengo.Ensemble(n_neurons=50, dimensions=1)
    def find_peak(x):
        return domain[np.argmax(fs.reconstruct(x))]
    nengo.Connection(ens, peak, function=find_peak)
    
    probe_peak = nengo.Probe(peak, synapse=0.03)
    
sim = nengo.Simulator(model)
sim.run(0.2)
pylab.plot(sim.trange(), sim.data[probe_peak])
pylab.show()

model = nengo.Network()
with model:
    stim = nengo.Node(gaussian(mag=1, mean=0.5, sd=0.1))
    ens = nengo.Ensemble(n_neurons=100, dimensions=2000,
                         encoders=space,
                         eval_points=space,
                         radius=radius
                        )
    nengo.Connection(stim, ens)
    probe_func = nengo.Probe(ens, synapse=0.03)
    probe_spikes = nengo.Probe(ens.neurons)
    
sim = nengo.Simulator(model)
sim.run(0.2)
pylab.imshow(sim.data[probe_func], extent=(-1,1,0.2,0), aspect=10.0)
pylab.ylabel('time')
pylab.figure()
pylab.hist(np.mean(sim.data[probe_spikes], axis=0))
pylab.xlabel('Hz')
pylab.show()

radius = np.mean(np.linalg.norm(space.sample(10), axis=1))

model = nengo.Network()
with model:
    stim = nengo.Node(gaussian(mag=1, mean=0.5, sd=0.1))
    ens = nengo.Ensemble(n_neurons=100, dimensions=2000,
                         encoders=space,
                         eval_points=space.sample(5000)/radius,
                         radius=radius
                        )
    nengo.Connection(stim, ens)
    probe_func = nengo.Probe(ens, synapse=0.03)
    probe_spikes = nengo.Probe(ens.neurons)
    
sim = nengo.Simulator(model)
sim.run(0.2)
pylab.imshow(sim.data[probe_func], extent=(-1,1,0.2,0), aspect=10.0)
pylab.ylabel('time')
pylab.figure()
pylab.hist(np.mean(sim.data[probe_spikes], axis=0))
pylab.xlabel('Hz')
pylab.show()

fs = nengo.FunctionSpace(
        nengo.dists.Function(gaussian,
                             mag=nengo.dists.Uniform(0, 1),
                             mean=nengo.dists.Uniform(-1, 1),
                             sd=nengo.dists.Uniform(0.1, 0.3)), 
        n_basis=20)

model = nengo.Network()
with model:
    stim = nengo.Node(fs.project(gaussian(mag=1, mean=0.5, sd=0.1)))    
    ens = nengo.Ensemble(n_neurons=500, dimensions=fs.n_basis)
    ens.encoders = fs.project(
                        nengo.dists.Function(gaussian,
                        mean=nengo.dists.Uniform(-1, 1),
                        sd=0.1,
                        mag=1))
    ens.eval_points = fs.project(fs.space)
    nengo.Connection(stim, ens)
    probe_func = nengo.Probe(ens, synapse=0.03)
sim = nengo.Simulator(model)
sim.run(0.2)
pylab.plot(domain, fs.reconstruct(sim.data[probe_func][-1]))
pylab.figure()
pylab.imshow(fs.reconstruct(sim.data[probe_func]), extent=(-1,1,0.2,0), aspect=10.0)
pylab.ylabel('time')
pylab.show()

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=500, dimensions=fs.n_basis)
    ens.encoders = fs.project(fs.space)
    ens.eval_points = fs.project(fs.space)
    
    # input
    stim = fs.make_stimulus_node(gaussian, 3)
    nengo.Connection(stim, ens)    
    stim_control = nengo.Node([1, 0, 0.2])
    nengo.Connection(stim_control, stim)
    
    #output
    plot = fs.make_plot_node(domain=domain)
    nengo.Connection(ens, plot, synapse=0.1)
    
from nengo_gui.ipython import IPythonViz
IPythonViz(model, cfg='funcspace.cfg')

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=500, dimensions=fs.n_basis)
    ens.encoders = fs.project(space)
    ens.eval_points = fs.project(fs.space)
    
    # input
    stim = fs.make_stimulus_node(gaussian, 3)
    nengo.Connection(stim, ens)    
    stim_control = nengo.Node([1, 0, 0.2])
    nengo.Connection(stim_control, stim)
    
    #output
    plot = fs.make_plot_node(domain=domain, n_pts=50, lines=2)
    nengo.Connection(ens, plot[:fs.n_basis], synapse=0.1)
    nengo.Connection(stim, plot[fs.n_basis:], synapse=0.1)
    
from nengo_gui.ipython import IPythonViz
IPythonViz(model, cfg='funcspace2.cfg')

domain = np.random.uniform(-1, 1, size=(1000, 2))

def gaussian2d(meanx, meany, sd):
    return np.exp(-((domain[:,0]-meanx)**2+(domain[:,1]-meany)**2)/(2*sd**2))

fs = nengo.FunctionSpace(
        nengo.dists.Function(gaussian2d,
                             meanx=nengo.dists.Uniform(-1, 1), 
                             meany=nengo.dists.Uniform(-1, 1), 
                             sd=nengo.dists.Uniform(0.1, 0.7)),
        
        n_basis=50)

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=500, dimensions=fs.n_basis)
    ens.encoders = fs.project(fs.space)
    ens.eval_points = fs.project(fs.space)
    
    stimulus = nengo.Node(fs.project(gaussian2d(0,0,0.2)))
    nengo.Connection(stimulus, ens)
    
    probe = nengo.Probe(ens, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(0.2)

from mpl_toolkits.mplot3d import Axes3D
fig = pylab.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(domain[:,0], domain[:,1], fs.reconstruct(sim.data[probe][-1]))
pylab.show()

domain = np.linspace(-1, 1, 2000)

def gaussian(mag, mean, sd):
    return mag * np.exp(-(domain-mean)**2/(2*sd**2))

fs = nengo.FunctionSpace(
        nengo.dists.Function(gaussian,
                             mag=nengo.dists.Uniform(0, 1),
                             mean=nengo.dists.Uniform(-1, 1),
                             sd=nengo.dists.Uniform(0.1, 0.3)), 
        n_basis=40)

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=2000, dimensions=fs.n_basis+1)
    ens.encoders = nengo.dists.Combined(
                        [fs.project(fs.space), nengo.dists.UniformHypersphere(surface=True)], 
                        [fs.n_basis,1])    
    ens.eval_points = nengo.dists.Combined(
                        [fs.project(fs.space), nengo.dists.UniformHypersphere(surface=False)], 
                        [fs.n_basis,1])

    stim_shift = nengo.Node([0])
    nengo.Connection(stim_shift, ens[-1])
    
    # input
    stim = fs.make_stimulus_node(gaussian, 3)
    nengo.Connection(stim, ens[:-1])    
    stim_control = nengo.Node([1, 0, 0.13])
    nengo.Connection(stim_control, stim)
    
    #output
    plot = fs.make_plot_node(domain=domain)
    
    def shift_func(x):
        shift = int(x[-1]*500)
        return fs.project(np.roll(fs.reconstruct(x[:-1]), shift))
    
    nengo.Connection(ens, plot, synapse=0.1, function=shift_func)
    
from nengo_gui.ipython import IPythonViz
IPythonViz(model, cfg='funcspace3.cfg')



