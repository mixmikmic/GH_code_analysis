reload(cn)

import analysis
import connectivity as cn

import networkx as nx

fear = nx.read_graphml("Fear199_backup.graphml")

# Doing the spectral embedding
fear_se = cn.spec_clust(fear, 3)
print fear_se.shape

se_regions, norm_dict, se_regions_nodes = cn.get_dict_real(fear, fear_se, 'Fear199_regions.csv')

c, o_c, con_graph = cn.get_connectivity_hard(se_regions, norm_dict, 0.02)

# Getting the connectivity adjacency matrix
con_adj_mat = nx.adj_matrix(con_graph).todense()
print(con_adj_mat.shape)
print(con_adj_mat)

fig = cn.plot_con_mat(con_adj_mat, output_path=None, show=True)

print(con_graph.nodes()[300])

print(con_graph.node[590])

c, o_c, con_graph = cn.get_connectivity_hard(se_regions, norm_dict, 0.01)
# Getting the connectivity adjacency matrix
con_adj_mat = nx.adj_matrix(con_graph).todense()
print(con_adj_mat.shape)
# print(con_adj_mat)
fig = cn.plot_con_mat(con_adj_mat, output_path=None, show=True)

c, o_c, con_graph = cn.get_connectivity_hard(se_regions, norm_dict, 0.015)
# Getting the connectivity adjacency matrix
con_adj_mat = nx.adj_matrix(con_graph).todense()
print(con_adj_mat.shape)
# print(con_adj_mat)
fig = cn.plot_con_mat(con_adj_mat, output_path=None, show=True)



