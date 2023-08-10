import pandas
from neo4j.v1 import GraphDatabase
import hetio.readwrite

from hetmech.diffusion import diffuse

url = 'https://github.com/dhimmel/hetionet/raw/76550e6c93fbe92124edc71725e8c7dd4ca8b1f5/hetnet/json/hetionet-v1.0.json.bz2'
graph = hetio.readwrite.read_graph(url)
metagraph = graph.metagraph

# MetaGraph node/edge count
metagraph.n_nodes, metagraph.n_edges

# Graph node/edge count
graph.n_nodes, graph.n_edges

# Uses the official neo4j-python-driver. See https://github.com/neo4j/neo4j-python-driver

query = '''
MATCH (disease:Disease)-[assoc:ASSOCIATES_DaG]-(gene:Gene)
WHERE disease.name = 'epilepsy syndrome'
RETURN
 gene.name AS gene_symbol,
 gene.description AS gene_name,
 gene.identifier AS entrez_gene_id,
 assoc.sources AS sources
ORDER BY gene_symbol
'''

driver = GraphDatabase.driver("bolt://neo4j.het.io")
with driver.session() as session:
    result = session.run(query)
    gene_df = pandas.DataFrame((x.values() for x in result), columns=result.keys())

gene_df.head()

epilepsy_genes = list()
for entrez_gene_id in gene_df.entrez_gene_id:
    node_id = 'Gene', entrez_gene_id
    node = graph.node_dict.get(node_id)
    if node:
        epilepsy_genes.append(node)
len(epilepsy_genes)

metapath = metagraph.metapath_from_abbrev('GiGpBP')
source_node_weights = {gene: 1 for gene in epilepsy_genes}
pathway_scores = diffuse(graph, metapath, source_node_weights, column_damping=1, row_damping=1)
rows = [(pathway.name, score) for pathway, score in pathway_scores.items()]
target_df = pandas.DataFrame(rows, columns=['target_node', 'score'])
target_df = target_df.sort_values('score', ascending=False)

len(target_df)

sum(target_df.score)

metapath

target_df.head()

