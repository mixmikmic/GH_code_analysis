from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import time

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
PID_Raw = pd.read_csv(wd+'Network_Data_Raw/PathwayCommons9.pid.hgnc.sif',sep='\t', header=-1)
print 'Raw interactions in NCI PID:', PID_Raw.shape[0]

# Filter all interactions that contain a CHEBI: item
PID_filt = PID_Raw[(~PID_Raw[0].str.contains(':')) & (~PID_Raw[2].str.contains(':'))]
PID_edgelist = PID_filt[[0, 2]].values.tolist()
print 'Protein-Protein interactions in NCI PID:', len(PID_edgelist)

# Sort each edge representation for filtering
PID_edgelist_sorted = [sorted(edge) for edge in PID_edgelist]

# Filter edgelist for duplicate nodes and for self-edges
PID_edgelist_filt = gct.filter_converted_edgelist(PID_edgelist_sorted)

# Save genelist to file
outdir = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/Network_SIFs_Symbol/'
gct.write_edgelist(PID_edgelist_filt, outdir+'PID_Symbol.sif')

