import pandas as pd
from network_evaluation_tools import gene_conversion_tools as gct

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
HumanInteractome_Raw = pd.read_csv(wd+'Network_Data_Raw/HI-II-14.tsv',sep='\t')
print 'Raw Interactions in HI-II-14:', len(HumanInteractome_Raw)

# Convert table of interactions to edgelist (no scores given)
# Also no gene symbol conversion necessary because network is given in symbol format already
HumanInteractome_edgelist = HumanInteractome_Raw[['Symbol A', 'Symbol B']].values.tolist()
print 'Edges in HI-II-14:', len(HumanInteractome_edgelist)

# Sort each edge representation for filtering
HumanInteractome_edgelist_sorted = [sorted(edge) for edge in HumanInteractome_edgelist]

# Filter edgelist for duplicate nodes and for self-edges
HumanInteractome_edgelist_filt = gct.filter_converted_edgelist(HumanInteractome_edgelist_sorted)

# Save genelist to file
outdir = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/Network_SIFs_Symbol/'
gct.write_edgelist(HumanInteractome_edgelist_filt, outdir+'HumanInteractome_Symbol.sif')

