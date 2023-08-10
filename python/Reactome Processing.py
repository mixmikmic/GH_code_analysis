from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import itertools
import time

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
Reactome_Raw = pd.read_csv(wd+'Network_Data_Raw/Reactome_v60.interactions.txt',sep='\t',skiprows=1, header=-1, low_memory=False)
print 'Raw Edges in Reactome v60:', len(Reactome_Raw)

# Get edge list of network (filter for duplicate edges and self-edges)
query_edgelist_filt = Reactome_Raw[[0,3]].drop_duplicates()
print len(query_edgelist_filt), "Raw Reactome Edges after removing duplicate edges"
query_edgelist_filt2 = query_edgelist_filt[query_edgelist_filt[0]!=query_edgelist_filt[3]]
print len(query_edgelist_filt2), "Raw Reactome Edges after removing duplicate and self-edges"
query_edgelist = query_edgelist_filt2.values.tolist()

# Extract gene list
Reactome_Raw_Genes = list(set(query_edgelist_filt2[0]).union(set(query_edgelist_filt2[3])))

query_string, valid_genes, invalid_genes = gct.query_constructor(Reactome_Raw_Genes)

# Set scopes (gene naming systems to search)
scopes = "uniprot"

# Set fields (systems from which to return gene names from)
fields = "symbol, entrezgene"

# Query MyGene.Info
match_list = gct.query_batch(query_string, scopes=scopes, fields=fields)
print len(match_list), 'Matched query results'

match_table_trim, query_to_symbol, query_to_entrez = gct.construct_query_map_table(match_list, valid_genes)

# Format edge list by removing prefixes from all interactors
query_edgelist_fmt = [[gct.get_identifier_without_prefix(edge[0]), gct.get_identifier_without_prefix(edge[1])] for edge in query_edgelist]

# Convert network edge list to symbol
Reactome_edgelist_symbol = gct.convert_edgelist(query_edgelist_fmt, query_to_symbol, weighted=False)

# Filter converted edge list
Reactome_edgelist_symbol_filt = gct.filter_converted_edgelist(Reactome_edgelist_symbol, weighted=False)

# Save filtered, converted edge list to file
gct.write_edgelist(Reactome_edgelist_symbol_filt, wd+'Network_SIFs_Symbol/Reactome_Symbol.sif', binary=True)



