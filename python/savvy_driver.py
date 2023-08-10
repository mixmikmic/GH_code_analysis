# comment out this import if you would like to see warnings
import warnings; warnings.filterwarnings('ignore')
import copy

from bokeh.plotting import show, output_notebook
import os.path as op
import os
import savvy.data_processing as dp
import savvy.interactive_plots as ip
from savvy.plotting import make_plot, make_second_order_heatmap
import savvy.network_tools as nt

output_notebook()

# read the sensitivity analysis files from sample data loaded on github:
datapath = op.join(os.getcwd(),'savvy/sample_data_files/')
sa_dict = dp.get_sa_data(datapath)

# Plot the first and total order indices for all files in "datapath"
ip.interact_with_plot_all_outputs(sa_dict)

# If you do not want to use the widget interactivity, but still want the ability
# to explore the outputs on tabs, you can use this function and manually pass
# all the arguments to makeplot

# ip.plot_all_outputs(sa_dict, min_val=0.01, top=30)

# Plot the second order plots with tabs for all the options
ip.plot_all_second_order(sa_dict, top=5, mirror=True)

sa_dict_net = copy.deepcopy(sa_dict)
g = nt.build_graph(sa_dict_net['sample-output1'], sens='ST', top=10, min_sens=0.01,
                   edge_cutoff=0.0)
nt.plot_network_circle(g, inline=True, scale=200)
# test = nt.plot_network_random(g, scale=200)

dp.find_unimportant_params('S1', 'savvy/sample_data_files/')
dp.find_unimportant_params('ST', 'savvy/sample_data_files/');

# demo of making the 1st and total order sensitivity index plot
df = sa_dict['sample-output1'][0]
p = make_plot(df, lgaxis=True, minvalues=0.0, top=30, stacked=True,
              errorbar=True, showS1=True, showST=True)
show(p)

# demo of the basic second order sensitivity index heat map
df2 = sa_dict['sample-output1'][1]
incl_lst = ['Tmax', 'Carbon', 'Hydrogen', 'k38', 'k48', 'k34']
s = make_second_order_heatmap(df2, top=3, mirror=True, name='demo',
                              include=incl_lst)
show(s)



