import redis

r = redis.StrictRedis(host='localhost')

r.set('key', 'value')
print r.get('key')

import nengo
import numpy as np

r = redis.StrictRedis(host='localhost')
model1 = nengo.Network()
with model1:
    stim = nengo.Node(np.sin)
    a = nengo.Ensemble(100, 1)
    output = nengo.Node(lambda t, x: r.set('decoded_value', x[0]), size_in=1)
    nengo.Connection(stim, a)    
    nengo.Connection(a, output)
    
import nengo_gui.ipython
nengo_gui.ipython.IPythonViz(model1, 'model1.cfg')

import nengo
import numpy as np

r2 = redis.StrictRedis(host='localhost')

model2 = nengo.Network()
with model2:
    reader = nengo.Node(lambda t: float(r2.get('decoded_value')))
    a = nengo.Ensemble(100, 1)
    nengo.Connection(reader, a)    
    
import nengo_gui.ipython
nengo_gui.ipython.IPythonViz(model2, 'model2.cfg')



