import networkx as nx
import pandas as pd

G = nx.read_gpickle('major_us_cities')

G.nodes(data=True)

# Initialize the dataframe, using the nodes as the index
df = pd.DataFrame(index=G.nodes())

df['location'] = pd.Series(nx.get_node_attributes(G, 'location'))
df['population'] = pd.Series(nx.get_node_attributes(G, 'population'))

df.head()

df['clustering'] = pd.Series(nx.clustering(G))
df['degree'] = pd.Series(G.degree())

df

G.edges(data=True)

# Initialize the dataframe, using the edges as the index
df = pd.DataFrame(index=G.edges())

df['weight'] = pd.Series(nx.get_edge_attributes(G, 'weight'))

df

df['preferential attachment'] = [i[2] for i in nx.preferential_attachment(G, df.index)]

df

df['Common Neighbors'] = df.index.map(lambda city: len(list(nx.common_neighbors(G, city[0], city[1]))))

df



