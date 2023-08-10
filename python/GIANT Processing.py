from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import time

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
GIANT_Raw = pd.read_csv(wd+'/Network_Data_Raw/GIANT_All_Tissues_Top', sep='\t', header=-1, low_memory=False)
GIANT_Raw.columns = ['NodeA', 'NodeB', 'Prob']
print 'GIANT All Tissues (Top) Interactions:', GIANT_Raw.shape[0]

# Get all genes to convert from GeneMANIA
GIANT_Raw_Genes = list(set(GIANT_Raw['NodeA']).union(GIANT_Raw['NodeB']))
# Convert all entrezIDs to string forst
GIANT_Raw_Genes = [str(entrezID) for entrezID in GIANT_Raw_Genes]

query_string, valid_genes, invalid_genes = gct.query_constructor(GIANT_Raw_Genes)

# Set scopes (gene naming systems to search)
scopes = "entrezgene, retired, alias"

# Set fields (systems from which to return gene names from)
fields = "symbol, entrezgene"

# Query MyGene.Info
match_list = gct.query_batch(query_string, scopes=scopes, fields=fields)
print len(match_list), 'Matched query results'

match_table_trim, query_to_symbol, query_to_entrez = gct.construct_query_map_table(match_list, valid_genes)

GIANT_Raw_edgelist = GIANT_Raw.values.tolist()

# Convert GIANT network edgelist
GIANT_Raw_edgelist_symbol = [sorted([query_to_symbol[str(int(edge[0]))], query_to_symbol[str(int(edge[1]))]])+[edge[2]] for edge in GIANT_Raw_edgelist]

# Filter GIANT network edgelist
GIANT_edgelist_symbol_filt = gct.filter_converted_edgelist(GIANT_Raw_edgelist_symbol, remove_self_edges=True, weighted=True)

GIANT_edgelist_symbol_filt_table = pd.DataFrame(GIANT_edgelist_symbol_filt, columns=['NodeA', 'NodeB', 'Score'])

# Filter edges by score quantile
q_score = GIANT_edgelist_symbol_filt_table['Score'].quantile(0.9)
print '90% score:', q_score
GIANTtop_edgelist = GIANT_edgelist_symbol_filt_table[GIANT_edgelist_symbol_filt_table['Score']>q_score]

# Save weighted network for GIANT filtered to top 10% of downloaded edges to file
GIANTtop_edgelist.to_csv('/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/Network_SIFs_Symbol/GIANT_Symbol.sif', sep='\t', header=False, index=False)

# Create filtered network for GIANT
GIANT90_edgelist = dit.filter_weighted_network_sif(wd+'Network_SIFs_Symbol/GIANT_Symbol.sif', nodeA_col=0, nodeB_col=1, score_col=2, 
                                                   q=0.9, delimiter='\t', verbose=True, save_path=wd+'Network_SIFs_Symbol/GIANT90_Symbol.sif')



