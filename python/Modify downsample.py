from clustergrammer import Network
from clustergrammer_widget import clustergrammer_widget
net = Network()

net.load_file('txt/rc_two_cats.txt')

df = net.export_df()

net.load_df(df)

ds_data = net.downsample(ds_type='kmeans', axis='row', num_samples=10)

net.make_clust()



clustergrammer_widget(network=net.widget())

type(ds_data)

ds_data = list(ds_data)

len(ds_data)

# ds_datad

net.load_file('txt/rc_two_cats.txt')
df = net.export_df()

rows = df.index.tolist()
cols = df.columns.tolist()

rows[0]

new_rows = []
i = 0
for inst_row in rows:
#     inst_tuple = ('cluster', 'category '+ str(i))
    inst_tuple= 'cluster'
    i = i + 1
    new_rows.append(inst_tuple)
    
new_rows[0:3]

new_cols = []
i = 0
for inst_col in cols:
    inst_tuple = ('col', 'category '+ str(i))
#     inst_tuple= 'col'
    i = i + 1
    new_cols.append(inst_tuple)
    
new_cols[0:3]

df.columns = new_cols

df.index = new_rows





net.load_df(df)

net.dat['nodes']

net.make_clust()

clustergrammer_widget(network=net.widget())



