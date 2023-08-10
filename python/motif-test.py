import graph_tool as gt
from graph_tool.all import *
import motifwalk as mw

from motifwalk import utils as u

from motifwalk.motifs import Motif

from motifwalk.motifs import *

from motifwalk.motifs import analysis

analysis.count_motif()

g = random_graph(1000, lambda: (5,5))
result = motifs(g, m.num_vertices(), motif_list=[m], return_maps=True)

mo, count, vmap = result

len(vmap[0])

p = vmap[0][15]

graph = p.get_graph()

gt.draw.graph_draw(graph, output_size=(100,100))

p[0]

p[1]

p[2]

g.get_out_edges(151)

g.get_out_edges(383)


g.get_out_edges(562)

m3_7.gt_motif.get_edges()

p.get_graph().get_edges()

p[0]

g.get_out_edges(151)

p.get_array()

m = m3_9.gt_motif
gt.draw.graph_draw(m, output_size=(200,200))

result = motifs(g, m.num_vertices(), motif_list=[m], return_maps=True)

mo, count, vmap = result

p = vmap[0][39]

p.get_array()

m.get_edges()

re = motifs(g, k=4, return_maps=True)

len(re)

len(re[1])

gt.draw.graph_draw(re[0][43], output_size=(100,200))
gt.draw.graph_draw(re[0][42], output_size=(100,200))
gt.draw.graph_draw(re[0][40], output_size=(100,200))
gt.draw.graph_draw(re[0][41], output_size=(100,200))

re[0][3].get_edges()[:,0:2]

[i for i in map(tuple, re[0][3].get_edges()[:,0:2])]

for i, motifs in enumerate(re[0]):
    arr = str([i for i in map(tuple, motifs.get_edges()[:,0:2])])
    s = "m4_{} = Motif({}, is_directed=True, name='m4_{}')".format(i, arr, i)
    print(s)

la = Graph()

na = nx.Graph()

na.size()

from itertools import combinations

[i for i in combinations(p.get_array(), 2)]

def construct_motif_graph(graph_container, vertex_maps=None, motif=None):
    """Construct and return a undirected gt graph containing
    motif relationship. TODO: Add anchors nodes

    Parameters:
    graph_container - GraphContainer - Store the original network
    vertex_map - list - contains PropertyMap that maps to vertices in motif

    Returns:
    m_graph - gt.Graph - Undirected graph for motif cooccurence
    """
    if motif is not None and motif.anchors is not None:
        print("Warning: TODO refactor anchor code.")
    # graph_tool.Graph
    m_graph = Graph(directed=False)
    if vertex_maps is None:
        _, _, vertex_maps = count_motif(graph_container, motif)
    for prop in vertex_maps:
        edges = [i for i in combinations(prop.get_array(), 2)]
        m_graph.add_edge_listadd_edges_from(edges)
    return m_graph

la = construct_motif_graph(None, vertex_maps=re[2][0])

graph_draw(m4_0.gt_motif, output_size=(100,100))

graph_draw(m3_5.gt_motif, output_size=(100,100))

m3_5_r = motifs(g,k=len(m3_5.gt_motif.get_vertices()),motif_list=[m3_5.gt_motif],return_maps=True)

m3_5_r[2][0][0].get_array()

g.get_out_edges(216)

g.get_out_edges(938)

re = m3_5_r[2][0][0]

re.shrink_to_fit()

re[0]

re = isomorphism(re.get_graph(), m3_5.gt_motif, isomap=True)

re[1][2]

re = m3_5_r[2][0][0]

graph_draw(re.get_graph(), output_size=(100,100))

re.get_graph().get_edges()

re[0]

re[1]

re[2]

g.get_out_edges(216)

re.get_graph().get_edges()

re[0], re[1], re[2]

for i in _:
    print(g.get_out_edges(i))

# select some vertices
vfilt = g.new_vertex_property('bool');
vfilt[216] = True
vfilt[756] = True
vfilt[938] = True

sub = GraphView(g, vfilt)

ka = isomorphism(sub, m3_5.gt_motif, isomap=True)

ka[1][216], ka[1][756], ka[1][938]

[i for i in [216, 756, 938] if ka[1][i] in {0,1}]



