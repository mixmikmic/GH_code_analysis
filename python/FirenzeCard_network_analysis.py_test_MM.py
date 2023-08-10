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

edges = make_dynamic_firenze_card_edgelist(df)
edges.head()

static = make_static_firenze_card_edgelist(edges)
static.head()

g = make_firenze_card_static_graph(static, nodes)
ig.summary(g)

# g = delete_paired_edges(g)
# ig.summary(g)

mat = make_origin_destination_matrix(g)
mat.head()

plot_origin_destination_matrix_heatmap(mat)

























paths = make_firenze_card_daily_paths(df)
paths.head()

agg = aggregate_firenze_card_daily_paths(paths)
agg.head()

plot_aggregate_firenze_card_daily_paths(agg)

plot_firenze_card_static_graph(g)











# from_to_home = from_to_home_firenze_card_edges_generator(paths, nodes)

# from_to_home.reset_index(inplace=True)
# supp_edges = pd.DataFrame({'from': ['start'] * from_to_home.shape[0] + from_to_home['short_name'].tolist(),
#                            'to': from_to_home['short_name'].tolist() + ['end'] * from_to_home.shape[0],
#                            'weight': from_to_home['home_to_node'].tolist() + from_to_home['node_to_home'].tolist()})
# supp_edges.dropna(how='any', inplace=True)
# static2 = pd.concat([static, supp_edges])
# # static2 = static2[static2['from'] != 'source']

# static2[(static2['from']=='source')].sort_values('weight',ascending=False)

# static2[(static2['from']=='start')].sort_values('weight',ascending=False)

# static2 = add_home_node_firenze_card_static_edgelist(from_to_home, static)



















df1 = fill_out_time_series(df,timeunitname='hour',timeunitcode='h',start_date='2016-06-01',end_date='2016-10-01')
df1.head()

df1.columns = ['short_name','hour','total_people']
time_series_full_plot(df1)

plot_frequencies_total(df)

df2 = df.groupby('museum_id').sum()[
    ['total_adults', 'minors']]  # Take only these two columns. Could use df1, but might as well go back to df.
df2['total_people'] = df2['total_adults'] + df2[
    'minors']  # Again, add them. Note that we don't delete these columns, so they will be plotted as well.
df2.sort_values('total_people', inplace=True, ascending=False)  # Sort the values for easy viewing
df2.plot.bar(figsize=(16, 8))
plt.title('Number of Firenze card visitors')
plt.xlabel('Museum')
plt.ylabel('Number of people')
# plt.yscale('log')
plt.show()



