from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import time

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
PC_Raw = pd.read_csv(wd+'Network_Data_Raw/PathwayCommons9.All.hgnc.sif', sep='\t', header=-1)
print 'Raw interactions in Pathway Commons v9:', PC_Raw.shape[0]

# Filter all interactions that contain a CHEBI: item
PC_filt = PC_Raw[(~PC_Raw[0].str.contains(':')) & (~PC_Raw[2].str.contains(':'))]
PC_edgelist = PC_filt[[0, 2]].values.tolist()
print 'Protein-Protein interactions in Pathway Commons v9:', len(PC_edgelist)

# Sort each edge representation for filtering
PC_edgelist_sorted = [sorted(edge) for edge in PC_edgelist]

# Filter edgelist for duplicate nodes and for self-edges
PC_edgelist_filt = gct.filter_converted_edgelist(PC_edgelist_sorted)

# Save genelist to file
outdir = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/Network_SIFs_Symbol/'
gct.write_edgelist(PC_edgelist_filt, outdir+'PathwayCommons_Symbol.sif')

