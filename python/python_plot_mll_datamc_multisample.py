import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

base_dir = "/home/olivito/datasci/spark/clustertest/dimuonReduced_180123_153653/"
input_path_mll = base_dir + "mll_merged.parquet"
input_path_counts = base_dir + "count_merged.parquet"
df_mll = pq.ParquetDataset(input_path_mll).read().to_pandas()
df_counts = pq.ParquetDataset(input_path_counts).read().to_pandas().set_index('sampleID')

print len(df_mll)
df_mll.tail()

print df_counts['count'][101]

lumi = 11.6 # in fb-1, for 2012B+C
df_counts['weight'] = lumi * 1000. * df_counts['xsec'] / df_counts['count']
# reset the data samples to have weight = 1.  not used anyway
df_counts['weight'][100] = 1
df_counts['weight'][101] = 1
df_counts

def get_weight(id):
    return df_counts['weight'][id]

# this is very slow... there's probably a much faster way to do this
df_mll['weight'] = df_mll['sampleID'].apply(get_weight)

df_mll.head()

# map of sampleID numbers to samples for the final plot

samples_dict = OrderedDict()
samples_dict['W+jets'] = [11,12,13]
samples_dict['WW'] = [5]
samples_dict['ZZ'] = [8,9,10]
samples_dict['WZ'] = [6,7]
samples_dict['single top'] = [3,4]
samples_dict['ttbar'] = [1,2]
samples_dict['Drell-Yan'] = [0]

samples_dfs = OrderedDict()
for sample,IDs in samples_dict.items():
    samples_dfs[sample] = df_mll.loc[df_mll['sampleID'].isin(IDs)]
    
# treat data separately
df_data = df_mll.loc[df_mll['sampleID'].isin([100,101])]

color_dict = OrderedDict()
color_dict['W+jets'] = 'gray'
color_dict['WW'] = 'cyan'
color_dict['ZZ'] = 'magenta'
color_dict['WZ'] = 'red'
color_dict['single top'] = 'yellow'
color_dict['ttbar'] = 'green'
color_dict['Drell-Yan'] = 'blue'

fig = plt.figure(figsize=(8,6))

# set up binning.  get also bin centers, to use for data points
bin_size = 2
xmin = 50
xmax = 131 # should make this slightly larger than the desired end point of the plot
binning = np.arange(xmin,xmax,bin_size)
binning_centers = np.arange(xmin + bin_size/2.,xmax,bin_size)

# for MC, can plot all together as a stacked histogram
plt.hist([df['mll'] for df in samples_dfs.values()],
         weights=[df['weight'] for df in samples_dfs.values()],
         label=samples_dfs.keys(),
         color=color_dict.values(),
         bins=binning, histtype='bar', stacked=True
        )

# for data, get the bin counts and plot with error bars
hist_data,_ = np.histogram(df_data['mll'],binning)
errors_data = np.sqrt(hist_data)
plt.errorbar(binning_centers, hist_data, errors_data,
             linestyle='none', marker='o', color='black', label='Data, 2012BC'
            )

# add axis labels and legend
plt.xlabel('$M_{\mu\mu}$ [GeV]',fontsize='large')
plt.ylabel('Events / '+str(bin_size)+' GeV',fontsize='large')
ax = plt.gca() # "get current axis"
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), fontsize='large')  # reverse to keep order consistent

#fig.savefig('mll_multisample_lin.png')

plt.yscale('log')
fig.savefig('mll_multisample_log.png')

plt.show()



