get_ipython().run_line_magic('pylab', 'inline')
import networkx as nx
import numpy as np

def setup(n_agents=100, WS_k=10, WS_p=0.05, n_diffusands=100, n_thresholds=7):
    # setup network
    np.random.seed()
    g = nx.watts_strogatz_graph(n_agents, WS_k, p=WS_p)
    nx.set_node_attributes(g, 'diffusands', {n:np.zeros((n_diffusands, n_thresholds)) for n in range(n_agents)})
    
    # seed diffusands
    for diffusand in range(n_diffusands):
        #seed = int(np.random.uniform(0,n_agents))
        seed = diffusand # give each diffusand a unique starting node
        g.node[seed]['diffusands'][diffusand,:] = 1  # the seed is infected
        for neighbor in g[seed]:
            g.node[neighbor]['diffusands'][diffusand,:]=1  # and so are all his neighbors

    return g

g = setup()

def draw(g):
    n_plots = g.node[0]['diffusands'].shape[1]
    plt.figure(figsize=(2*n_plots, 3))
    for i in range(n_plots):
        plt.subplot(1,n_plots,i+1)
        adopters = np.array([g.node[n]['diffusands'][:,i] for n in g])
        plt.imshow(adopters, vmin=0, vmax=1)
        plt.xlabel('diffusand')
        if i == 0:
            plt.ylabel('agent')
        else:
            plt.yticks([])
        plt.title('Threshold %i' % (i+1))
draw(g)

def simulate(g, n_steps = 100):
    for step in range(n_steps):
        for n in np.random.permutation(g):
            exposures = np.array([g.node[neighbor]['diffusands'] for neighbor in g[n]]).sum(axis=0)
            g.node[n]['diffusands'] = exposures > np.arange(exposures.shape[1])
            
    return g

simulate(g) 
draw(g)

def measure(g):
    res = dict()
    res['penetration'] = np.array([g.node[n]['diffusands'] for n in g]).sum(axis=0)
    res['penetration fraction'] = res['penetration']/nx.number_of_nodes(g)
    res['interesting'] = ((res['penetration'] != np.tile(res['penetration'][:,0], 
                                                         (res['penetration'].shape[1], 1)).T) *
                           (res['penetration'] != 0))
    res['percent interesting cases'] = np.mean(res['interesting'])
    res['interesting cascade sizes'] = res['penetration'][res['interesting']]
    res['mean interesting cascade size'] = res['interesting cascade sizes'].mean()
    return res

res = measure(g)
plt.plot(np.ones_like(res['penetration']).cumsum(axis=1),
         res['penetration'], 'bo', alpha=.5)
plt.xlabel('Threshold')
plt.ylabel('Penetration');

interesting_cascade_sizes = []
sizes = np.arange(100, 10000, 100)
for n_agents in sizes:
    g = setup(n_agents=n_agents, n_diffusands=100)
    simulate(g)
    res = measure(g)
    interesting_cascade_sizes.append(res['interesting cascade sizes'])

for i, size in enumerate(sizes):
    plt.plot([size]*len(interesting_cascade_sizes[i]), interesting_cascade_sizes[i], 'bo', alpha=.25)
    
plt.ylabel('Extent of interesting cascade')
plt.xlabel('Size of population');

for i, size in enumerate(sizes):
    plt.plot([size]*len(interesting_cascade_sizes[i]), interesting_cascade_sizes[i]/size, 'bo', alpha=.25)
    
plt.ylabel('Fraction of population participating\nin an interesting cascade')
plt.xlabel('Size of population');

interesting_cascade_sizes = []
degrees = np.arange(4, 30, 2)
for degree in degrees:
    g = setup(n_agents=1000, n_diffusands=100, WS_k=degree)
    simulate(g)
    res = measure(g)
    interesting_cascade_sizes.append(res['interesting cascade sizes'])

degrees

for i, degree in enumerate(degrees):
    plt.plot([degree]*len(interesting_cascade_sizes[i]), interesting_cascade_sizes[i], 'bo', alpha=.25)
    
plt.ylabel('Extent of interesting cascade')
plt.xlabel('Degree');

interesting_cascade_sizes = []
sizes = np.arange(100, 10000, 100)
for n_agents in sizes:
    g = setup(n_agents=n_agents, WS_k=4, WS_p=.2)
    simulate(g)
    res = measure(g)
    interesting_cascade_sizes.append(res['interesting cascade sizes'])

for i, size in enumerate(sizes):
    plt.plot([size]*len(interesting_cascade_sizes[i]), interesting_cascade_sizes[i]/size, 'bo', alpha=.25)
    
plt.ylabel('Fraction of population participating\nin an interesting cascade')
plt.xlabel('Size of population');



