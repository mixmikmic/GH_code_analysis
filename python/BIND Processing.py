import pandas as pd
from network_evaluation_tools import gene_conversion_tools as gct

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
BIND_Raw = pd.read_csv(wd+'Network_Data_Raw/PathwayCommons.8.bind.BINARY_SIF.hgnc.txt.sif',sep='\t', header=-1)

# Convert table of interactions to edgelist (no scores given)
# Also no gene symbol conversion necessary because network is given in symbol format already
BIND_edgelist = BIND_Raw[[0, 2]].values.tolist()
print 'Edges in BIND:', len(BIND_edgelist)

# Sort each edge representation for filtering
BIND_edgelist_sorted = [sorted(edge) for edge in BIND_edgelist]

# Filter edgelist for duplicate nodes and for self-edges
BIND_edgelist_filt = gct.filter_converted_edgelist(BIND_edgelist_sorted)

# Save genelist to file
outdir = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/Network_SIFs_Symbol/'
gct.write_edgelist(BIND_edgelist_filt, outdir+'BIND_Symbol.sif')



