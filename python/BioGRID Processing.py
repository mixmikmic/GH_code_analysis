from network_evaluation_tools import gene_conversion_tools as gct
import pandas as pd
import itertools

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
BioGRID_Raw = pd.read_csv(wd+'Network_Data_Raw/BioGRID/BIOGRID-ORGANISM-3.4.149.tab2/BIOGRID-ORGANISM-Homo_sapiens-3.4.149.tab2.txt',sep='\t', low_memory=False)
print 'Raw edge count in BioGRID:', len(BioGRID_Raw)

# Show not all interactions in BioGRID are physical PPI, though the overwhelming majority are
BioGRID_Raw['Experimental System Type'].value_counts()

# Not all interactions are from Human
BioGRID_Raw['Organism Interactor A'].value_counts().head()

# Not all interactions are from Human
BioGRID_Raw['Organism Interactor B'].value_counts().head()

BioGRID_Human_Only = BioGRID_Raw[(BioGRID_Raw['Organism Interactor A']==9606) & (BioGRID_Raw['Organism Interactor B']==9606)]
print 'Human-Human only interactions in BioGRID 3.4.149:', len(BioGRID_Human_Only)

# Any missing symbol names in column A?
BioGRID_Human_Only['Official Symbol Interactor A'][BioGRID_Human_Only['Official Symbol Interactor A']=='-']

# Any missing symbol names in column B?
BioGRID_Human_Only['Official Symbol Interactor B'][BioGRID_Human_Only['Official Symbol Interactor B']=='-']

# Convert table of interactions to edgelist (no scores given)
# Also no gene symbol conversion necessary because network is given in symbol format already
BioGRID_edgelist = BioGRID_Human_Only[['Official Symbol Interactor A', 'Official Symbol Interactor B']].values.tolist()
print 'Edges in BioGRID:', len(BioGRID_edgelist)

# Sort each edge representation for filtering
BioGRID_edgelist_sorted = [sorted(edge) for edge in BioGRID_edgelist]

# Filter edgelist for duplicate nodes and for self-edges
BioGRID_edgelist_filt = gct.filter_converted_edgelist(BioGRID_edgelist_sorted)

# Save genelist to file
outdir = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/Network_SIFs_Symbol/'
gct.write_edgelist(BioGRID_edgelist_filt, outdir+'BioGRID_Symbol.sif')

