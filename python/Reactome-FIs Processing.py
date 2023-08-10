from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import itertools
import time

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
Reactome_FIs_Raw = pd.read_csv(wd+'Network_Data_Raw/FIsInGene_022717_with_annotations.txt',sep='\t')
print 'Raw edges in ReactomeFI:', Reactome_FIs_Raw.shape[0]

# Extract gene list
Reactome_FIs_Raw_Genes = list(set(Reactome_FIs_Raw['Gene1']).union(set(Reactome_FIs_Raw['Gene2'])))

# Find "invalid genes" by text format
query_string, valid_genes, invalid_genes = gct.query_constructor(Reactome_FIs_Raw_Genes, exclude_prefixes=['CHEBI'], print_invalid_genes=True)

# Get Edgelist of network
query_edgelist = Reactome_FIs_Raw[['Gene1','Gene2', 'Score']].values.tolist()

# Filter query edges
query_edgelist_filt = gct.filter_query_edgelist(query_edgelist,invalid_genes)

# Filter edge list
ReactomeFI_edgelist_filt = gct.filter_converted_edgelist(query_edgelist_filt, weighted=True)

# Save filtered, converted edge list to file
gct.write_edgelist(ReactomeFI_edgelist_filt, wd+'Network_SIFs_Symbol/ReactomeFI_Symbol.sif', binary=False)

# Create filtered network
ReactomeFI90_edgelist = dit.filter_weighted_network_sif(wd+'Network_SIFs_Symbol/ReactomeFI_Symbol.sif', nodeA_col=0, nodeB_col=1, score_col=2, 
                                                        q=0.9, delimiter='\t', verbose=True, save_path=wd+'Network_SIFs_Symbol/ReactomeFI90_edgelist_Symbol.sif')

# The filter function didn't work here because the max value makes up >90% of the edges. 
# We need to filter but keep all max edges instead
ReactomeFI_edgelist = pd.DataFrame(ReactomeFI_edgelist_filt, columns=['NodeA', 'NodeB', 'Score'])
q_score = ReactomeFI_edgelist['Score'].quantile(0.9)
ReactomeFI_edgelist_filt2 = ReactomeFI_edgelist[ReactomeFI_edgelist['Score']>=q_score]
print ReactomeFI_edgelist_filt2.shape[0], '/', ReactomeFI_edgelist.shape[0], 'edges kept, ', float(ReactomeFI_edgelist_filt2.shape[0])/ReactomeFI_edgelist.shape[0]

# Essentially >85% of the edges have the 'maximum score' which makes almost no sense for filtering further

