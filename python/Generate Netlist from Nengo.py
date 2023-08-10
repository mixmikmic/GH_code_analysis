get_ipython().magic('matplotlib inline')
import pylab
import seaborn
import numpy as np
import pandas

import netlist
reload(netlist)
import pprint

import ctn_benchmark.spa.memory
model = ctn_benchmark.spa.memory.SemanticMemory().make_model(D=1)

# extract the graph from a nengo model.  This returns:
#   ensembles    (groups of x->A->y neurons)
#   nodes        (non-neural components used for organization)
#   connections  (edges)
e, n, c = netlist.compute_graph(model)

# compute some basic stats
pprint.pprint(netlist.calc_stats(e, n, c))

# draw the graph
netlist.plot_graph(e, n, c, size=(10,10))

import ctn_benchmark.spa.memory
model = ctn_benchmark.spa.memory.SemanticMemory().make_model(D=1)

e, n, c = netlist.compute_graph(model)
netlist.simplify_conns(e, n, c)
pprint.pprint(netlist.calc_stats(e, n, c))

netlist.plot_graph(e, n, c, size=(10,10))

import ctn_benchmark.spa.memory

stats = []
Ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
for D in Ds:
    model = ctn_benchmark.spa.memory.SemanticMemory().make_model(D=D)

    e, n, c = netlist.compute_graph(model)
    netlist.simplify_conns(e, n, c)
    s = netlist.calc_stats(e, n, c)
    s['D'] = D
    stats.append(s)

df = pandas.DataFrame(stats)
df

pylab.semilogy(df['memory'], label='memory')
pylab.semilogy(df['neurons'], label='neurons')
pylab.semilogy(df['messages'], label='messages')
pylab.semilogy(df['values'], label='values')
pylab.xticks(np.arange(len(Ds)), Ds)
pylab.legend(loc='best')
pylab.xlabel('size of semantic memory')
pylab.show()

model = ctn_benchmark.spa.memory.SemanticMemory().make_model(D=512)
e, n, c = netlist.compute_graph(model)

ens = {}
for ee in e.values():
    key = (ee['size_in'], ee['n_neurons'], ee['size_out'])
    ens[key] = ens.get(key, 0) + 1
pprint.pprint(ens)

model = ctn_benchmark.nengo.SPASequence().make_model(n_actions=5)
e, n, c = netlist.compute_graph(model)
netlist.simplify_conns(e, n, c)
pprint.pprint(netlist.calc_stats(e, n, c))

netlist.plot_graph(e, n, c, size=(10,10))

import ctn_benchmark.spa.memory

stats = []
actions = [5, 10, 20, 50, 100, 200]
for n_actions in actions:
    model = ctn_benchmark.nengo.SPASequence().make_model(n_actions=n_actions, D=512)
    e, n, c = netlist.compute_graph(model)
    netlist.simplify_conns(e, n, c)
    s = netlist.calc_stats(e, n, c)
    s['n_actions'] = n_actions
    stats.append(s)

df = pandas.DataFrame(stats)
df

pylab.semilogy(df['memory'], label='memory')
pylab.semilogy(df['neurons'], label='neurons')
pylab.semilogy(df['messages'], label='messages')
pylab.semilogy(df['values'], label='values')
pylab.xticks(np.arange(len(actions)), actions)
pylab.legend(loc='best')
pylab.xlabel('number of actions to choose from')
pylab.show()

model = model = ctn_benchmark.nengo.SPASequence().make_model(n_actions=200, D=512)
e, n, c = netlist.compute_graph(model)

ens = {}
for ee in e.values():
    key = (ee['size_in'], ee['n_neurons'], ee['size_out'])
    ens[key] = ens.get(key, 0) + 1
pprint.pprint(ens)

import shelve
db = shelve.open('spaun_netlist')
e = db['e']
n = db['n']
c = db['c']
db.close()

netlist.calc_stats(e, n, c)

ens = {}
for ee in e.values():
    key = (ee['size_in'], ee['n_neurons'], ee['size_out'])
    ens[key] = ens.get(key, 0) + 1
pprint.pprint(ens)



