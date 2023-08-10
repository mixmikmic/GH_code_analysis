from network_evaluation_tools import gene_conversion_tools as gct
import pandas as pd
import itertools

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
BioPlex_Raw = pd.read_csv(wd+'Network_Data_Raw/BioPlex_interactionList_v4a.tsv',sep='\t')
print 'Raw edge count in BioPlex:', len(BioPlex_Raw)

BioPlex_Raw.head()

# Convert table of interactions to edgelist (no scores given)
# Also no gene symbol conversion necessary because network is given in symbol format already
BioPlex_edgelist = BioPlex_Raw[['SymbolA', 'SymbolB']].values.tolist()
print 'Edges in BIND:', len(BioPlex_edgelist)

# Sort each edge representation for filtering
BioPlex_edgelist_sorted = [sorted(edge) for edge in BioPlex_edgelist]

# Filter edgelist for duplicate nodes and for self-edges
BioPlex_edgelist_filt = gct.filter_converted_edgelist(BioPlex_edgelist)

# Write network to file
gct.write_edgelist(BioPlex_edgelist_filt, wd+'Network_SIFs_Symbol/BioPlex_Symbol.sif', binary=True)

