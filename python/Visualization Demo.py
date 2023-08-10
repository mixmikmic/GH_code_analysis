import os

from pybel import from_pickle
from pybel_tools.selection import get_subgraph_by_annotation_value
from pybel_tools.visualization import to_jupyter

graph_path = os.path.expanduser('~/dev/bms/aetionomy/alzheimers/alzheimers.gpickle')

graph = from_pickle(graph_path)
wnt_subgraph = get_subgraph_by_annotation_value(graph, annotation='Subgraph', value='Wnt signaling subgraph')
to_jupyter(wnt_subgraph)

