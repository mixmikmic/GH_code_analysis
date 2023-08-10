from network_evaluation_tools import gene_conversion_tools as gct
from network_evaluation_tools import data_import_tools as dit
import pandas as pd
import time

# Load and filter STRING for only human-human protein interactions
wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
starttime=time.time()
g=open(wd+'Network_Data_Raw/STRING/STRING_human_v10.5.txt','w')
with open(wd+'Network_Data_Raw/STRING/protein.links.v10.5.txt') as f:
    for line in f:
        edge = line.split(' ')
        if edge[0].startswith('9606') and edge[1].startswith('9606'):
            g.write(edge[0].split('.')[1]+'\t'+edge[1].split('.')[1]+'\t'+edge[2]+'\n')
print 'Filtered human-human STRING interactions only:', time.time()-starttime, 'seconds'
g.close()

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
STRING_Raw = pd.read_csv(wd+'Network_Data_Raw/STRING/STRING_human_v10.5.txt',sep='\t',header=-1)
STRING_Raw.columns = ['NodeA', 'NodeB', 'Score']
print 'Raw Edges in STRING v10.5:', len(STRING_Raw)

STRING_Raw_filt = STRING_Raw.drop_duplicates()
print 'Edges in STRING v10.5 after dropping duplicates:', len(STRING_Raw_filt)

STRING_Genes = list(set(STRING_Raw_filt['NodeA']).union(set(STRING_Raw_filt['NodeB'])))

query_edgelist = STRING_Raw_filt[['NodeA', 'NodeB', 'Score']].values.tolist()

query_string, valid_genes, invalid_genes = gct.query_constructor(STRING_Genes)

# Set scopes (gene naming systems to search)
scopes = "ensemblprotein"

# Set fields (systems from which to return gene names from)
fields = "symbol, entrezgene"

# Query MyGene.Info
match_list = gct.query_batch(query_string, scopes=scopes, fields=fields)
print len(match_list), 'Matched query results'

match_table_trim, query_to_symbol, query_to_entrez = gct.construct_query_map_table(match_list, valid_genes)

get_ipython().run_cell_magic('time', '', '# Convert weighted edge list\nSTRING_edgelist_symbol = gct.convert_edgelist(query_edgelist, query_to_symbol, weighted=True)')

# Filter converted edge list
STRING_edgelist_symbol_filt = gct.filter_converted_edgelist(STRING_edgelist_symbol, weighted=True)

# Write network to file
gct.write_edgelist(STRING_edgelist_symbol_filt, wd+'Network_SIFs_Symbol/STRING_Symbol.sif', binary=False)

# Create filtered network
STRING90_edgelist = dit.filter_weighted_network_sif(wd+'Network_SIFs_Symbol/STRING_Symbol.sif', nodeA_col=0, nodeB_col=1, score_col=2, 
                                                    q=0.9, delimiter='\t', verbose=True, save_path=wd+'Network_SIFs_Symbol/STRING90_Symbol.sif')



