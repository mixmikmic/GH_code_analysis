get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pandas as pd
import networkx as nx
import pandas as pd

plt.figure(figsize=(4,4))
A_f_As = np.linspace(0,1,100)
B_f_As = np.linspace(0,1,100)
f_A = .6

A_f_Ag, B_f_Ag = np.meshgrid(A_f_As, B_f_As)
measures = f_A * A_f_Ag + (1-f_A) * B_f_Ag

levels = np.arange(0,1,.1)
CS = plt.contourf(A_f_Ag, B_f_Ag, measures, levels )
plt.clabel(CS, inline=1, fontsize=10, colors='k')
plt.vlines(f_A, 0, 1)
plt.title('%.02f Vote A' % f_A)
plt.xlabel('A voters belief')
plt.ylabel('B voters belief');

plt.figure(figsize=(10,4))
A_f_As = np.linspace(0,1,100)
B_f_As = np.linspace(0,1,100)

A_f_Ag, B_f_Ag = np.meshgrid(A_f_As, B_f_As)


for i, f_A in enumerate(np.arange(.1, 1, .1)):
    plt.subplot(2,5,i+1)
    measures = f_A * A_f_Ag + (1-f_A) * B_f_Ag

    levels = [0, f_A, 1]
    CS = plt.contourf(A_f_Ag, B_f_Ag, measures, levels, vmin=0, vmax=f_A )
    plt.vlines(f_A, 0, 1)
    plt.title('%.02f Vote A' % f_A)
    plt.xlabel('A voters belief')
    plt.ylabel('B voters belief')
plt.tight_layout()

def setup(n_voters=1000, WS_k=4, WS_p=.1, 
          f_know=.1,  # what fraction of the population is given the ground truth
          initial_bias_towards=.3,  # for the remaining (uninformed) population, how likely are they to believe?
          a=0, # how much should knowledge be clustered? 0 is uniform dist...
         ):
    np.random.seed()
    g = nx.watts_strogatz_graph(n_voters, k=WS_k, p=WS_p)
    
    #cluster knowledge
    x = np.linspace(0,1,n_voters)
    y = x**a*(1-x)**a
    ps = y/sum(y)
    
    knowers = np.random.choice(range(n_voters), size=int(n_voters*f_know), replace=False, p=ps)
    have_info = [1if i in knowers else 0 for i in range(n_voters) ]
    nx.set_node_attributes(g, name='informed', values={n:i for n, i in enumerate(have_info)})

    believe = where(have_info, 1, np.random.binomial(1, initial_bias_towards, n_voters))
    nx.set_node_attributes(g, name='believe', values={n:i for n, i in enumerate(believe)})

    return g

g = setup()

def simulate(g, threshold=.5, max_rounds=10, include_own=False):
    f_A = np.mean([g.node[n]['believe'] for n in g])
    for i in range(max_rounds):
        for n in np.random.permutation(g):
            if g.node[n]['informed'] == 0:
                neighbors_beliefs = np.array([g.node[nb]['believe'] for nb in g[n]])
                g.node[n]['believe'] = 1 if np.mean(neighbors_beliefs) > threshold else 0
                
        new_f_A = np.mean([g.node[n]['believe'] for n in g])
        if new_f_A == f_A:  # converged
            print('Converged at %.02f after %i rounds' % (f_A, i+1))
            break
        else:
            f_A = new_f_A

    # individuals form an expectation for the fraction of their neighbors who believe
    for n in g:
        neighbors_beliefs = [g.node[nb]['believe'] for nb in g[n]]
        if include_own:
            neighbors_beliefs.append(g.node[n]['believe'])
            #print(neighbors_beliefs)
        g.node[n]['expectation believe'] = np.mean(np.array(neighbors_beliefs))
    
    return g

simulate(g, include_own=True)

def measure(g):
    f_A = np.mean([g.node[n]['believe'] for n in g])
    A_f_A = np.mean([g.node[n]['expectation believe'] for n in g if g.node[n]['believe']==1])
    B_f_A = np.mean([g.node[n]['expectation believe'] for n in g if g.node[n]['believe']==0])
    
    return f_A, A_f_A, B_f_A

f_A, A_f_A, B_f_A = measure(g)

plt.figure(figsize=(4,4))

def plot(f_A, A_f_A, B_f_A):
    A_f_As = np.linspace(0,1,100)
    B_f_As = np.linspace(0,1,100)

    A_f_Ag, B_f_Ag = np.meshgrid(A_f_As, B_f_As)
    measures = f_A * A_f_Ag + (1-f_A) * B_f_Ag

    levels = [-0.01, f_A, 1.01]
    CS = plt.contourf(A_f_Ag, B_f_Ag, measures, levels, vmin=0, vmax=f_A )

    plt.vlines(f_A, 0, 1)
    plt.hlines(f_A, 0, 1)

    plt.plot(A_f_A, B_f_A, 'k*')

    plt.title('%.02f Vote A' % f_A)
    plt.xlabel('A voters belief')
    plt.ylabel('B voters belief');
    
plot(f_A, A_f_A, B_f_A)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    g = setup(WS_p=np.random.uniform(0,.3), 
              f_know=np.random.uniform(0,.5), 
              initial_bias_towards=np.random.uniform(0,.4))
    g = simulate(g, max_rounds=0)
    f_A, A_f_A, B_f_A = measure(g)
    
    plot(f_A, A_f_A, B_f_A)
    
plt.tight_layout()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    g = setup(WS_p=np.random.uniform(0,.3), 
              f_know=np.random.uniform(0,.5), 
              initial_bias_towards=np.random.uniform(0,.4), a=10)
    g = simulate(g, max_rounds=0)
    f_A, A_f_A, B_f_A = measure(g)
    
    plot(f_A, A_f_A, B_f_A)
    
plt.tight_layout()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    g = setup(WS_p=np.random.uniform(0,.3), 
              f_know=np.random.uniform(0,.5), 
              initial_bias_towards=np.random.uniform(0,.4), a=10)
    g = simulate(g, max_rounds=10)
    f_A, A_f_A, B_f_A = measure(g)
    
    plot(f_A, A_f_A, B_f_A)
    
plt.tight_layout()

def setup2(n_voters=1000, WS_k=6, WS_p=.1,
          f_know=.1,  # what fraction of the population is given the ground truth
          initial_bias_towards=.3,  # for the remaining (uninformed) population, how likely are they to believe?
          a=0, # how much should knowledge be clustered? 0 is uniform dist...
          b=0,
         ):
    np.random.seed()
    g = nx.nx.powerlaw_cluster_graph(n_voters, WS_k, WS_p)
    
    #cluster knowledge
    x = np.linspace(0,1,n_voters)
    y = x**a * (1-x)**b
    ps = y/sum(y)
    
    knowers = np.random.choice(range(n_voters), size=int(n_voters*f_know), replace=False, p=ps)
    have_info = [1 if i in knowers else 0 for i in range(n_voters) ]
    nx.set_node_attributes(g, name='informed', values={n:i for n, i in enumerate(have_info)})

    believe = where(have_info, 1, np.random.binomial(1, initial_bias_towards, n_voters))
    nx.set_node_attributes(g, name='believe', values={n:i for n, i in enumerate(believe)})

    return g


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    g = setup2(WS_p=np.random.uniform(.6,1), 
              f_know=np.random.uniform(0,.5), 
              initial_bias_towards=np.random.uniform(0,.4))
    g = simulate(g, max_rounds=0)
    f_A, A_f_A, B_f_A = measure(g)
    
    plot(f_A, A_f_A, B_f_A)
    
plt.tight_layout()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    g = setup2(WS_p=np.random.uniform(.6,1), 
              f_know=np.random.uniform(0,.5), 
              initial_bias_towards=np.random.uniform(0,.4))
    g = simulate(g, max_rounds=10)
    f_A, A_f_A, B_f_A = measure(g)
    
    plot(f_A, A_f_A, B_f_A)
    
plt.tight_layout()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    g = setup2(WS_p=np.random.uniform(.6,1), 
              f_know=np.random.uniform(0,.5), 
              initial_bias_towards=np.random.uniform(0,.4), b=3)
    g = simulate(g, max_rounds=0)
    f_A, A_f_A, B_f_A = measure(g)
    
    plot(f_A, A_f_A, B_f_A)
    
plt.tight_layout()

a = 0,
b = 3
plt.subplot(2,1,1)
plt.plot(g, list(dict(nx.degree(g)).values()))

plt.ylabel('Degree')

x = np.linspace(0,1,n_voters)
y = x**a * (1-x)**b
plt.subplot(2,1,2)
plt.plot(x, y, 'r')
plt.xlabel('Node Number')
plt.ylabel('Likelihood of\nhaving information');

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    g = setup2(WS_p=np.random.uniform(.6,1), 
              f_know=np.random.uniform(0,.5), 
              initial_bias_towards=np.random.uniform(0,.4), a=3)
    g = simulate(g, max_rounds=0)
    f_A, A_f_A, B_f_A = measure(g)
    
    plot(f_A, A_f_A, B_f_A)
    
plt.tight_layout()

a = 3
b = 0
plt.subplot(2,1,1)
plt.plot(g, list(dict(nx.degree(g)).values()))

plt.ylabel('Degree')

x = np.linspace(0,1,n_voters)
y = x**a * (1-x)**b
plt.subplot(2,1,2)
plt.plot(x, y, 'r')
plt.xlabel('Node Number')
plt.ylabel('Likelihood of\nhaving information');



