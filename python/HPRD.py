from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
HPRD_Raw = pd.read_csv(wd+'Network_Data_Raw/HPRD_Release9_062910/BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt',sep='\t',header=-1)

# Assign column names from README file from archive
HPRD_Raw.columns = ['Interactor 1 Gene Symbol', 'Interactor 1 HPRD ID', 'Interactor 1 RefSeq ID',
                    'Interactor 2 Gene Symbol', 'Interactor 2 HPRD ID', 'Interactor 2 RefSeq ID',
                    'Experiment Type', 'PubMed ID']

# Convert table of interactions to edgelist (no scores given)
# Also no gene symbol conversion necessary because network is given in symbol format already
HPRD_edgelist = HPRD_Raw[['Interactor 1 Gene Symbol', 'Interactor 2 Gene Symbol']].values.tolist()
print 'Edges in HPRD:', len(HPRD_edgelist)

# Sort each edge representation for filtering
HPRD_edgelist_sorted = [sorted(edge) for edge in HPRD_edgelist]

# Filter edgelist for duplicate nodes and for self-edges
HPRD_edgelist_filt = gct.filter_converted_edgelist(HPRD_edgelist_sorted)

# Save genelist to file
outdir = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/Network_SIFs_Symbol/'
gct.write_edgelist(HPRD_edgelist_filt, outdir+'HPRD_Symbol.sif')



