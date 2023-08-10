# How to compute intercept range given an angle range

import numpy as np
angle_range_degrees = np.array([15.0, 35.0])
angle_range = angle_range_degrees * np.pi / 180

print np.cos(angle_range)

import nengo
model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=400, dimensions=2,
                         intercepts=nengo.dists.Uniform(0.81, 0.97),
                         )
    
sim = nengo.Simulator(model)

import numpy as np
theta_degrees = np.linspace(-100, 100, 201)  # in degrees
theta = theta_degrees * np.pi / 180

x = np.vstack([np.sin(theta), np.cos(theta)]).T

response_curves = np.zeros((ens.n_neurons, len(theta)))

inputs, activity = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=x)
    
    

get_ipython().magic('matplotlib inline')
import pylab

pylab.plot(theta_degrees, activity[:,:20])
pylab.xlabel('represented angle')
pylab.ylabel('firing rate (Hz)')
pylab.show()



