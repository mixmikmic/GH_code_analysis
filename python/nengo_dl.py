get_ipython().magic('matplotlib inline')
import pylab
import numpy as np
import nengo
import nengo_dl
import tensorflow as tf

D = 2

n_batches = 1
inputs = np.random.normal(size=(n_batches, 50, D))

targets = np.product(inputs, axis=2)
targets.shape = targets.shape[0], targets.shape[1], 1

model = nengo.Network(seed=2)
with model:
    input = nengo.Node(nengo.processes.WhiteSignal(period=20, high=5, rms=0.5), size_out=D)
    l1 = nengo.Ensemble(n_neurons=100*D, dimensions=D, neuron_type=nengo.RectifiedLinear())
    output = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.RectifiedLinear())
    
    nengo.Connection(input, l1)
    nengo.Connection(l1, output, function=lambda x: np.product(x))
    
    p_out = nengo.Probe(output)
    p_in = nengo.Probe(input)
    

sim = nengo_dl.Simulator(model)

sim.run(2)
pylab.plot(sim.trange(), sim.data[p_out])
pylab.plot(sim.trange(), np.product(sim.data[p_in], axis=1), ls='--')
pylab.show()

sim.loss(inputs={input: inputs}, targets={p_out: targets}, objective='mse')


sim.train(inputs={input: inputs}, targets={p_out: targets}, 
          optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
          n_epochs=2)

sim.run(2)
pylab.plot(sim.trange(), sim.data[p_out])
pylab.plot(sim.trange(), np.product(sim.data[p_in], axis=1), ls='--')
pylab.show()

sim.loss(inputs={input: inputs}, targets={p_out: targets}, objective='mse')

