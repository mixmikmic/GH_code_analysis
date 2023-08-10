get_ipython().magic('matplotlib inline')
import nengo
import numpy as np
import pylab

model = nengo.Network()
with model:
    def stim_a_func(t):
        return np.sin(t*2*np.pi)
    stim_a = nengo.Node(stim_a_func)
    a = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim_a, a)
    
    def stim_b_func(t):
        return np.cos(t*np.pi)
    stim_b = nengo.Node(stim_b_func)
    b = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim_b, b)
    
    c = nengo.Ensemble(n_neurons=200, dimensions=2, radius=1.5)
    nengo.Connection(a, c[0])
    nengo.Connection(b, c[1])
    
    d = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    def multiply(x):
        return x[0] * x[1]
    nengo.Connection(c, d, function=multiply)

    data = []
    def output_function(t, x):
        data.append(x)
    output = nengo.Node(output_function, size_in=1)
    nengo.Connection(d, output, synapse=0.03)
    

sim = nengo.Simulator(model)
sim.run(2)

pylab.plot(data)

for ens in model.all_ensembles:
    print('Ensemble %d' % id(ens))
    print('    number of neurons: %d' % ens.n_neurons)
    print('    tau_rc: %g' % ens.neuron_type.tau_rc)
    print('    tau_ref: %g' % ens.neuron_type.tau_ref)
    print('    bias: %s' % sim.data[ens].bias)

for node in model.all_nodes:
    if node.size_out > 0:
        print('Input node %d' % id(node))
        print('Function to call: %s' % node.output)

for node in model.all_nodes:
    if node.size_in > 0:
        print('Output node %d' % id(node))
        print('Function to call: %s' % node.output)

for conn in model.all_connections:
    if isinstance(conn.pre_obj, nengo.Ensemble) and isinstance(conn.post_obj, nengo.Ensemble):
        print('Connection from %d to %d' % (id(conn.pre_obj), id(conn.post_obj)))
        print('    synapse time constant: %g' % conn.synapse.tau)        
        decoder = sim.data[conn].weights
        transform = nengo.utils.builder.full_transform(conn, allow_scalars=False)
        encoder = sim.data[conn.post_obj].scaled_encoders
        print('    decoder: %s' % decoder)
        print('    transform: %s' % transform)
        print('    encoder: %s' % encoder)

for conn in model.all_connections:
    if isinstance(conn.pre_obj, nengo.Node) and isinstance(conn.post_obj, nengo.Ensemble):
        print('Connection from input %d to %d' % (id(conn.pre_obj), id(conn.post_obj)))
        print('    synapse time constant: %g' % conn.synapse.tau)        
        transform = nengo.utils.builder.full_transform(conn, allow_scalars=False)
        encoder = sim.data[conn.post_obj].scaled_encoders
        print('    transform: %s' % transform)
        print('    encoder: %s' % encoder)

for conn in model.all_connections:
    if isinstance(conn.pre_obj, nengo.Ensemble) and isinstance(conn.post_obj, nengo.Node):
        print('Connection from %d to output %d' % (id(conn.pre_obj), id(conn.post_obj)))
        print('    synapse time constant: %g' % conn.synapse.tau)
        decoder = sim.data[conn].weights
        transform = nengo.utils.builder.full_transform(conn, allow_scalars=False)
        print('    decoder: %s' % decoder)
        print('    transform: %s' % transform)



