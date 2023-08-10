get_ipython().magic('matplotlib inline')
import pylab
import seaborn
import numpy as np
import nengo

class RegularSpiking(object):
    def __init__(self, size, dt=0.001):
        self.state = np.zeros(size)
        self.threshold = 1.0 / dt
        self.output = np.zeros(size)
    def __call__(self, t, x):
        self.output[:] = 0
        self.state += x
        spikes = np.where(self.state > self.threshold)
        self.output[spikes] = self.threshold
        self.state -= self.output
        return self.output

class PoissonSpikingApproximate(object):
    def __init__(self, size, seed, dt=0.001):
        self.rng = np.random.RandomState(seed=seed)
        self.dt = dt
        self.value = 1.0 / dt
        self.size = size
        self.output = np.zeros(size)
    def __call__(self, t, x):
        self.output[:] = 0
        p = 1.0 - np.exp(-x*self.dt)
        self.output[p>self.rng.rand(self.size)] = self.value
        return self.output

class PoissonSpikingExactBad(object):
    def __init__(self, size, seed, dt=0.001):
        self.rng = np.random.RandomState(seed=seed)
        self.dt = dt
        self.value = 1.0 / dt
        self.size = size
        self.output = np.zeros(size)
    def __call__(self, t, x):
        p = 1.0 - np.exp(-x*self.dt)
        self.output[:] = 0
        s = np.where(p>self.rng.rand(self.size))[0]
        self.output[s] += self.value
        count = len(s)
        while count > 0:
            s2 = np.where(p[s]>self.rng.rand(count))[0]
            s = s[s2]
            self.output[s] += self.value
            count = len(s)
            
        return self.output

class PoissonSpikingExact(object):
    def __init__(self, size, seed, dt=0.001):
        self.rng = np.random.RandomState(seed=seed)
        self.dt = dt
        self.value = 1.0 / dt
        self.size = size
        self.output = np.zeros(size)
    def next_spike_times(self, rate):        
        return -np.log(1.0-self.rng.rand(len(rate))) / rate
    def __call__(self, t, x):
        self.output[:] = 0
        
        next_spikes = self.next_spike_times(x)
        s = np.where(next_spikes<self.dt)[0]
        count = len(s)
        self.output[s] += self.value
        while count > 0:
            next_spikes[s] += self.next_spike_times(x[s])
            s2 = np.where(next_spikes[s]<self.dt)[0]
            count = len(s2)
            s = s[s2]
            self.output[s] += self.value
                
        return self.output

model = nengo.Network()
with model:
    freq=10
    stim = nengo.Node(lambda t: np.sin(t*np.pi*2*freq))
    ens = nengo.Ensemble(n_neurons=5, dimensions=1, neuron_type=nengo.LIFRate(), seed=1)
    nengo.Connection(stim, ens, synapse=None)
    
    regular_spikes = nengo.Node(RegularSpiking(ens.n_neurons), size_in=ens.n_neurons)
    nengo.Connection(ens.neurons, regular_spikes, synapse=None)

    poisson_spikes = nengo.Node(PoissonSpikingExact(ens.n_neurons, seed=1), size_in=ens.n_neurons)
    nengo.Connection(ens.neurons, poisson_spikes, synapse=None)
    
    p_rate = nengo.Probe(ens.neurons)
    p_regular = nengo.Probe(regular_spikes)
    p_poisson = nengo.Probe(poisson_spikes)
sim = nengo.Simulator(model)
sim.run(0.1)

pylab.figure(figsize=(10,8))

pylab.subplot(3,1,1)
pylab.plot(sim.trange(), sim.data[p_rate])
pylab.xlim(0, sim.time)
pylab.ylabel('rate')

pylab.subplot(3,1,2)
import nengo.utils.matplotlib
nengo.utils.matplotlib.rasterplot(sim.trange(), sim.data[p_regular])
pylab.xlim(0, sim.time)
pylab.ylabel('regular spiking')

pylab.subplot(3,1,3)
nengo.utils.matplotlib.rasterplot(sim.trange(), sim.data[p_poisson])
pylab.xlim(0, sim.time)
pylab.ylabel('poisson spiking')



pylab.show()

def test_accuracy(cls, rate, T=1):
    test_model = nengo.Network()
    with test_model:
        stim = nengo.Node(rate)
        spikes = nengo.Node(cls(1, seed=1), size_in=1)
        nengo.Connection(stim, spikes, synapse=None)

        p = nengo.Probe(spikes)
    sim = nengo.Simulator(test_model)
    sim.run(T, progress_bar=False)
    return np.mean(sim.data[p])

rates = np.linspace(0, 1000, 11)
result_approx = [test_accuracy(PoissonSpikingApproximate, r) for r in rates]
result_bad = [test_accuracy(PoissonSpikingExactBad, r) for r in rates]
result_exact = [test_accuracy(PoissonSpikingExact, r) for r in rates]

pylab.plot(rates, result_approx, label='spike rate approx')
pylab.plot(rates, result_bad, label='spike rate bad')
pylab.plot(rates, result_exact, label='spike rate exact')
pylab.plot(rates, rates, ls='--', c='k', label='ideal')
pylab.legend(loc='best')
pylab.show()



