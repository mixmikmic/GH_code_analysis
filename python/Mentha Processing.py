from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import itertools
import time

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
Mentha_Raw = pd.read_csv(wd+'Network_Data_Raw/mentha_2017_06_12', sep='\t', header=-1)
print 'Raw edge count in Mentha:', len(Mentha_Raw)

# Keep only human-human interactions
Mentha_Human_only = Mentha_Raw[(Mentha_Raw[9]=='taxid:9606(Homo sapiens)') & (Mentha_Raw[10]=='taxid:9606(Homo sapiens)')]
print 'Human-Human only interactions in Mentha:', len(Mentha_Human_only)

# Extract gene list
Human_Mentha_Genes = list(set(Mentha_Human_only[0]).union(set(Mentha_Human_only[1])))

# Construct list of genes to be submitted to MyGene.Info API (remove all genes with 'intact' prefix)
query_string, valid_genes, invalid_genes = gct.query_constructor(Human_Mentha_Genes)

# Set scopes (gene naming systems to search)
scopes = "uniprot"

# Set fields (systems from which to return gene names from)
fields = "symbol, entrezgene"

# Query MyGene.Info
match_list = gct.query_batch(query_string, scopes=scopes, fields=fields)
print len(match_list), 'Matched query results'

match_table_trim, query_to_symbol, query_to_entrez = gct.construct_query_map_table(match_list, valid_genes)

query_edgelist = Mentha_Human_only[[0, 1, 14]].drop_duplicates().values.tolist()

# Format edge list by removing 'uniprot:' prefix from all interactors
query_edgelist_fmt = [[gct.get_identifier_without_prefix(edge[0]), gct.get_identifier_without_prefix(edge[1]), float(edge[2].split(':')[-1])] for edge in query_edgelist]

# Convert network edge list to symbol
Mentha_edgelist_symbol = gct.convert_edgelist(query_edgelist_fmt, query_to_symbol, weighted=True)

# Filter converted edge list
Mentha_edgelist_symbol_filt = gct.filter_converted_edgelist(Mentha_edgelist_symbol, weighted=True)

# Save filtered, converted edge list to file
gct.write_edgelist(Mentha_edgelist_symbol_filt, wd+'Network_SIFs_Symbol/Mentha_Symbol.sif', binary=False)

# Create filtered network
Mentha90_edgelist = dit.filter_weighted_network_sif(wd+'Network_SIFs_Symbol/Mentha_Symbol.sif', nodeA_col=0, nodeB_col=1, score_col=2, 
                                                    q=0.9, delimiter='\t', verbose=True, save_path=wd+'Network_SIFs_Symbol/Mentha90_Symbol.sif')



