get_ipython().magic('matplotlib inline')
import networkx as nx

from logging import getLogger 
from eden.util import configure_logging
configure_logging(getLogger(''),verbosity=2)

def getpathgraph(labels):
    nlabels= len(labels)
    G=nx.path_graph(nlabels)
    for e in range(nlabels):
        G.node[e]['label']=labels[e]
    for e in range(nlabels-1):
        G.edge[e][e+1]['label']='.'
    return G


G=getpathgraph("ABC")

from eden.util import display
display.draw_graph(G, size=4, node_size=1500, prog='circo', size_x_to_y_ratio=3, font_size=11)

g1 = G.copy()

G=getpathgraph('ABBC')

from eden.util import display
display.draw_graph(G, size=4, node_size=1500, prog='circo', size_x_to_y_ratio=3, font_size=11)

g2 = G.copy()

G=getpathgraph('ABBBC')

from eden.util import display
display.draw_graph(G, size=4, node_size=1500, prog='circo', size_x_to_y_ratio=3, font_size=11)

g3 = G.copy()

G=getpathgraph('ABBBBC')


from eden.util import display
display.draw_graph(G, size=4, node_size=1500, prog='circo', size_x_to_y_ratio=3, font_size=11)

g4 = G.copy()

get_ipython().run_cell_magic('time', '', "import sys\nsys.path.append('..')\nimport graphlearn.graphlearn as gl\nfrom eden.converter.graph.gspan import gspan_to_eden\nimport itertools\ngr = [g1,g2,g3,g4,g4]\n\n\n\nsampler=gl.Sampler(radius_list=[0,1],thickness_list=[2],\n            min_cip_count=1,\n            min_interface_count=1)\nsampler.fit(gr,\n            grammar_n_jobs=-1)")

import graphlearn.utils.draw as draw
draw.draw_grammar(sampler.lsgg.productions,n_productions=None,
                     n_graphs_per_line=7, size=3, 
                     colormap='autumn', invert_colormap=True,
                     vertex_alpha=0.2, edge_alpha=0.2, node_size=380,
                     prog='circo', size_x_to_y_ratio=3)

draw.draw_grammar_stats(sampler.lsgg.productions)

#sample
draw.graphlearn(g1)
seed_graphs = [g1]
n_steps=50
res = sampler.sample(seed_graphs,
                        max_size_diff=-1,
                        n_samples=10,
                        batch_size=1,
                        n_steps=n_steps,
                        n_jobs=1)
#print 'asdasd',graphs
#draw
import matplotlib.pyplot as plt
scores=[]

for i,graphs in enumerate(list(res)):
    scores.append(sampler.monitors[i].sampling_info['score_history'])
    

    draw.draw_graph_set(graphs,
                           n_graphs_per_line=4, size=9,
                           prog='circo',
                           colormap='Paired', invert_colormap=False,node_border=0.5, vertex_color='_labels_',
                           vertex_alpha=0.5, edge_alpha=0.2, node_size=650)
    
for h in scores: plt.plot(h)
plt.show()

''' rewrite of directed sample happens later
import sys
sys.path.append('..')
import os
os.nice(19)
%matplotlib inline
import graphlearn.utils.draw as myutils
import graphlearn.directedsampler as ds
from eden.converter.graph.gspan import gspan_to_eden
import itertools
import matplotlib.pyplot as plt
import eden.graph as eg

steps=200
from logging import getLogger 
from eden.util import configure_logging
configure_logging(getLogger(''),verbosity=2)

#eg.Vectorizer(  normalization=False, inner_normalization=False, )

# so why do we fail if we use 10 here? 
vect=eg.Vectorizer( nbits= 17)


sampler= ds.directedSampler(thickness_list=[1],radius_list=[0,1,2], vectorizer=vect)


g1=getpathgraph('ABBBBBC')
food=[g1,g1,g1,g1,g1,g1]
target=getpathgraph('ABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBC')
target2=getpathgraph('ABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBC')
sampler.fit(food, n_jobs=1)


import scipy

graphs = sampler.sample(food,
                        #target_graph=target,
                        target_vector=   scipy.sparse.csr_matrix( sampler.get_average_vector([g1,target,target2])),
                        n_samples = 10,                   
                        n_steps=steps,
                        n_jobs=1,
                        select_cip_max_tries = 100,
                        same_core_size=False,
                        accept_annealing_factor=0,
                        accept_static_penalty= 1,
                        generatormode=False,
                        keep_duplicates=False,
                        probabilistic_core_choice=False
                        )


history=[]
for  i, gr in enumerate(graphs):
    print i,gr
    history.append(gr.graph['sampling_info']['score_history'])
    myutils.draw_graph_set_graphlearn(gr.graph['sampling_info']['graphs_history'],headlinehook=myutils.get_score_of_graph)
    
t = range(steps+1) 
for h in history[:3]:
    plt.plot(t, h)
plt.show()
t = range(steps+1) 
for h in history[3:6]:
    plt.plot(t, h)
plt.show()

print sampler.get_average_vector([g1,target,target2])
'''



