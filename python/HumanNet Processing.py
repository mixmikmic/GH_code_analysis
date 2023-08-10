from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import time

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
HumanNet_Raw = pd.read_csv(wd+'Network_Data_Raw/HumanNet.v1.join.txt',sep='\t',header=-1)

f = open(wd+'Network_Data_Raw/HumanNet.v1.evidence_code.txt')
HumanNet_headers = ['Gene 1', 'Gene 2']+[name.split(' = ')[0] for name in f.read().splitlines()[1:-1]]
HumanNet_Raw.columns = HumanNet_headers

# Extract gene list
HumanNet_Raw_Genes = list(set(HumanNet_Raw['Gene 1']).union(set(HumanNet_Raw['Gene 2'])))
HumanNet_Raw_Genes = [str(gene) for gene in HumanNet_Raw_Genes]

# Get edge list of network
query_edgelist = HumanNet_Raw[['Gene 1','Gene 2']].astype(str)
query_edgelist = pd.concat([query_edgelist, HumanNet_Raw['IntNet']], axis=1).values.tolist()
print len(query_edgelist), "HumanNet Edges"

query_string, valid_genes, invalid_genes = gct.query_constructor(HumanNet_Raw_Genes)

# Set scopes (gene naming systems to search)
scopes = "entrezgene, retired"

# Set fields (systems from which to return gene names from)
fields = "symbol, entrezgene"

# Query MyGene.Info
match_list = gct.query_batch(query_string, scopes=scopes, fields=fields)
print len(match_list), 'Matched query results'

match_table_trim, query_to_symbol, query_to_entrez = gct.construct_query_map_table(match_list, valid_genes)

get_ipython().run_cell_magic('time', '', '# Convert weighted edge list\nHumanNet_edgelist_symbol = gct.convert_edgelist(query_edgelist, query_to_symbol, weighted=True)')

# Filter converted edge list
HumanNet_edgelist_symbol_filt = gct.filter_converted_edgelist(HumanNet_edgelist_symbol, weighted=True)

# Write network to file
gct.write_edgelist(HumanNet_edgelist_symbol_filt, wd+'Network_SIFs_Symbol/HumanNet_Symbol.sif', binary=False)

# Create filtered network
HumanNet90_edgelist = dit.filter_weighted_network_sif(wd+'Network_SIFs_Symbol/HumanNet_Symbol.sif', nodeA_col=0, nodeB_col=1, score_col=2, 
                                                      q=0.9, delimiter='\t', verbose=True, save_path=wd+'Network_SIFs_Symbol/HumanNet90_Symbol.sif')

