from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import numpy as np

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
HINT_Bin_Raw = pd.read_csv(wd+'Network_Data_Raw/HINT_v4_binary_HomoSapiens.txt',sep='\t')
HINT_Com_Raw = pd.read_csv(wd+'Network_Data_Raw/HINT_v4_complex_HomoSapiens.txt',sep='\t')

HINT_Raw = pd.concat([HINT_Bin_Raw, HINT_Com_Raw])
print 'Concatenated list of edges:', HINT_Raw.shape
HINT_Raw = HINT_Raw.drop_duplicates()
print 'After duplicate edges removed:', HINT_Raw.shape

# Use UniProtID labels to annotate interactions
HPRD_Raw_Genes_Uniprot = set(HINT_Raw['Uniprot_A']).union(set(HINT_Raw['Uniprot_B']))

query_string, valid_genes, invalid_genes = gct.query_constructor(HPRD_Raw_Genes_Uniprot)

# Set scopes (gene naming systems to search)
scopes = "uniprot"

# Set fields (systems from which to return gene names from)
fields = "symbol, entrezgene"

# Query MyGene.Info
match_list = gct.query_batch(query_string, scopes=scopes, fields=fields)
print len(match_list), 'Matched query results'

match_table_trim, query_to_symbol, query_to_entrez = gct.construct_query_map_table(match_list, valid_genes)

HINT_edgelist = HINT_Raw[['Uniprot_A', 'Uniprot_B']].values.tolist()

# Convert edge list
HINT_edgelist_symbol = gct.convert_edgelist(HINT_edgelist, query_to_symbol, weighted=False)

# Filter edge list
HINT_edgelist_symbol_filt = gct.filter_converted_edgelist(HINT_edgelist_symbol)

# Save edge list
gct.write_edgelist(HINT_edgelist_symbol_filt, wd+'Network_SIFs_Symbol/HINT_Symbol.sif')

