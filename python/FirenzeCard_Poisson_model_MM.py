import os
import sys

sys.path.append('../../src/')
from utils.database import dbutils
from features.network_analysis import *

conn = dbutils.connect()
cursor = conn.cursor()

nodes = pd.read_sql('select * from optourism.firenze_card_locations', con=conn)
nodes.head()

firenzedata = pd.read_sql('select * from optourism.firenze_card_logs', con=conn)
firenzedata.head()

df = prepare_firenzedata(firenzedata, nodes)
df.head()
# firenzedata['date'] = pd.to_datetime(firenzedata['entry_time'],format='%Y-%m-%d %H:%M:%S').dt.date  # Convert the entry_time string to a datetime object

df['total_people'] = df['total_adults'] + df['minors']

edges = make_dynamic_firenze_card_edgelist(df)

static = make_static_firenze_card_edgelist(edges)

def make_firenze_card_static_graph_with_source(df,nodes,name='short_name',x='longitude',y='latitude'):
    """
    :param df: A static edgelist from above
    :param nodes: A data frame containing longitude and latitude
    :param name: the name on which to link the two data frames
    :param x: the longitude column name
    :param y: the latitude column name
    :return: an igraph graph object
    """
    g = ig.Graph.TupleList(df.itertuples(index=False), directed=True, weights=True)
    g.vs['indeg'] = g.strength(mode='in', weights='weight') # Save weighted indegree with dummy "source" node
#     g.delete_vertices([v.index for v in g.vs if v['name'] == u'source']) # Delete the dummy "source" node
    g.simplify(loops=False, combine_edges=sum) # Get rid of the few self-loops, which can plot strangely
    g.vs['label'] = g.vs["name"] # Names imported as 'name', but plot labels default to 'label'. Copy over.
    # Get coordinates, requires this lengthy query
    xy= pd.DataFrame({name: g.vs['label']}).merge(nodes[[name, x, y]], left_index=True, how='left', on=name)
    g.vs['x'] = (xy[x]).values.tolist()
    g.vs['y'] = (-1 * xy[y]).values.tolist() # Latitude is flipped, need to multiply by -1 to get correct orientation
    return g

g = make_firenze_card_static_graph_with_source(static,nodes)
ig.summary(g)

mat = make_origin_destination_matrix(g)

plot_origin_destination_matrix_heatmap(mat)

nodes.head()

temp = mat.sum(0).to_frame() # This will be "people leaving", used as an offset for a Poisson regression
temp.reset_index(inplace=True)
temp.columns = ['short_name','offset']
temp.head()
dfn = nodes.merge(temp, on='short_name')[['museum_id','short_name','offset']]
dfn.head()

temp = mat.sum(1).to_frame() # This will be "intrinsic popularity", total number of people entering the museum
temp.reset_index(inplace=True)
temp.columns = ['short_name','popularity']
temp.head()
dfn = dfn.merge(temp, on='short_name')[['museum_id','short_name','offset','popularity']]
dfn.head()

es = mat.stack().reset_index().rename(columns={'level_0':'source','level_1':'target', 0:'weight'})
es.head()

es.merge(dfn,)

