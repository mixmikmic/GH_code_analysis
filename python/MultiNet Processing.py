from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import itertools

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
MultiNet_Raw = pd.read_csv(wd+'Network_Data_Raw/Multinet.interactions.network_presence_2013_03_17.txt',sep='\t')
print 'Raw edge count in MultiNet:', MultiNet_Raw.shape[0]

# Build edge list from interaction column. The two parts of the interaction name on either side of '_' are gene symbols
MultiNet_edgelist = [interaction.split('_') for interaction in MultiNet_Raw['INTERACTION_NAME']]

# Sort each edge representation for filtering
MultiNet_edgelist_sorted = [sorted(edge) for edge in MultiNet_edgelist]

# Filter edgelist for duplicate nodes and for self-edges
MultiNet_edgelist_filt = gct.filter_converted_edgelist(MultiNet_edgelist_sorted)

# Save genelist to file
outdir = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/Network_SIFs_Symbol/'
gct.write_edgelist(MultiNet_edgelist_filt, outdir+'MultiNet_Symbol.sif')

