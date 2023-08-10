from jp_gene_viz.dGraph import clr
from jp_gene_viz import HMap
from jp_gene_viz import dGraph

# Heat map colors
HMap.HeatMap.max_clr = clr(222, 111, 0)
HMap.HeatMap.min_clr = clr(0, 222, 111)
HMap.HeatMap.zero_clr = clr(100, 100, 100)
# Network edge colors
dGraph.WGraph.positive_edge_color = clr(222, 111, 0)
dGraph.WGraph.negative_edge_color = clr(0, 222, 111)
dGraph.WGraph.zero_edge_color = clr(100, 100, 100)
# Network node colors
dGraph.WGraph.positive_node_color = clr(222, 111, 0)
dGraph.WGraph.negative_node_color = clr(0, 222, 111)
dGraph.WGraph.zero_node_color = clr(100, 100, 100)

from jp_gene_viz import LExpression

LExpression.load_javascript_support()

L = LExpression.LinkedExpressionNetwork()
L.load_network("network.tsv")
L.load_heatmap("expr.tsv")
L.network.rectangle_color = "#dddd55"
L.show()

L.network.rectangle_color = "yellow"



