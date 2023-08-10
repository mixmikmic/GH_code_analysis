import pandas
from neo4j.v1 import GraphDatabase
import hetio.readwrite
import hetio.neo4j
import hetio.pathtools

from hetmech.degree_weight import dwwc
from hetmech.matrix import get_node_to_position

url = 'https://github.com/dhimmel/hetionet/raw/76550e6c93fbe92124edc71725e8c7dd4ca8b1f5/hetnet/json/hetionet-v1.0.json.bz2'
graph = hetio.readwrite.read_graph(url)
metagraph = graph.metagraph

compound = 'DB00050'
disease = 'DOID:0050425'

damping_exponent = 0.4

# CbGeAlD does not contain duplicate nodes, so DWWC is equivalent to DWPC
metapath = metagraph.metapath_from_abbrev('CbGeAlD')

get_ipython().run_cell_magic('time', '', 'rows, cols, CbGeAlD_pc = dwwc(graph, metapath, damping=0)\nrows, cols, CbGeAlD_dwwc = dwwc(graph, metapath, damping=damping_exponent)')

CbGeAlD_dwwc.shape

# Density
CbGeAlD_dwwc.astype(bool).mean()

# Path count matrix
CbGeAlD_pc = CbGeAlD_pc.astype(int)
CbGeAlD_pc

# DWWC matrix
CbGeAlD_dwwc

i = rows.index(compound)
j = cols.index(disease)

# Path count
CbGeAlD_pc[i, j]

# degree-weighted walk count
CbGeAlD_dwwc[i, j]

query = hetio.neo4j.construct_dwpc_query(metapath, property='identifier')
print(query)

driver = GraphDatabase.driver("bolt://neo4j.het.io")
params = {
    'source': compound,
    'target': disease,
    'w': damping_exponent,
}
with driver.session() as session:
    result = session.run(query, params)
    result = result.single()
result

compound_id = 'Compound', 'DB00050'
disease_id = 'Disease', 'DOID:0050425'
paths = hetio.pathtools.paths_between(
    graph,
    source=graph.node_dict[compound_id],
    target=graph.node_dict[disease_id],
    metapath=metapath,
    duplicates=True,
)

paths

# Path count
len(paths)

# DWWC
hetio.pathtools.DWPC(paths, damping_exponent=damping_exponent)

