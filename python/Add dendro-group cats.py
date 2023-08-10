import pandas as pd
import numpy as np
from clustergrammer_widget import *
net = Network()

net.load_file('txt/rc_two_cats.txt')

dat_names = net.dat['node_info']['row']['full_names']

df = net.export_df()

df_names = df.index.tolist()

# nodes are in the same order in dat and df
for i in range(len(dat_names)):
    if dat_names[i] != df_names[i]:
        print('no match')



net.dat['node_info']['row'].keys()

net.make_clust()

net.dat['node_info']['row'].keys()

row_groups = net.dat['node_info']['row']['group']['05']

rows = df.index.tolist()

new_rows = []
for i in range(len(rows)):
    inst_row = rows[i]
    group_cat = 'group-' + str(row_groups[i])
    inst_row = inst_row + (group_cat,)
    new_rows.append(inst_row)

new_rows

df.index = new_rows

net.load_df(df)

net.make_clust()

clustergrammer_widget(network=net.widget())

net.dendro_cats('row', 5)

net.make_clust()

net.dendro_cats('row', 2)



clustergrammer_widget(network=net.widget())



