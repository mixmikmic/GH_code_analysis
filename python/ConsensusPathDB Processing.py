from network_evaluation_tools import gene_conversion_tools as gct
import re
import pandas as pd
import itertools
import time

wd = '/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/'
CPDB_Raw = pd.read_csv(wd+'Network_Data_Raw/ConsensusPathDB_human_PPI_v32',sep='\t',skiprows=1)
print CPDB_Raw.shape[0], 'raw interactions in ConsensusPathDB'

# Get all interaction from CPDB
CPDB_Raw_Interactions = list(CPDB_Raw['interaction_participants'])

# Remove self-edges from CPDB interactions
CPDB_Raw_Interactions_filt = []
for interaction in CPDB_Raw_Interactions:
    interaction_split = re.split(',|\.', interaction)
    if len(interaction_split) > 1:
        CPDB_Raw_Interactions_filt.append(interaction_split)

# Extract Binary interactions from lists of interactors (multi-protein complex interactions form cliques)
CPDB_binary_interactions = [list(itertools.combinations(gene_list, 2)) for gene_list in CPDB_Raw_Interactions_filt]
full_CPDB_interaction_list = list(itertools.chain(*CPDB_binary_interactions))
print 'Binary, non-self interactions in ConsensusPathDB v32:', len(full_CPDB_interaction_list)

# Load UniProt idmapping File
UniProt_ID_map = pd.read_csv('/cellar/users/jkhuang/Data/Projects/Network_Analysis/Data/HUMAN_9606_idmapping.dat', sep='\t', header=-1)

# Construct UniProtKB to UniProt Accession
UniProt_ID_map_filt = UniProt_ID_map[(UniProt_ID_map[1]=='UniProtKB-ID')]
UniProt_ID_map_filt = UniProt_ID_map_filt.set_index(2)
UniProtKB_ID_map = UniProt_ID_map_filt[0].to_dict()

# Convert CPDB interaction list to UniProt Accessions (if any interactions do not convert, do not keep the interaction)
CPDB_UniProtID_edgelist = []
for edge in full_CPDB_interaction_list:
    if (edge[0] in UniProtKB_ID_map) & (edge[1] in UniProtKB_ID_map):
        converted_edge = sorted([UniProtKB_ID_map[edge[0]], UniProtKB_ID_map[edge[1]]])
        CPDB_UniProtID_edgelist.append(converted_edge)
print len(CPDB_UniProtID_edgelist), 'interactions converted to UniProt Accession IDs'

# Remove self-edges and duplicate edges after conversion
CPDB_UniProtID_edgelist_filt = gct.filter_converted_edgelist(CPDB_UniProtID_edgelist)

CPDB_Converted_Genes = list(set(itertools.chain.from_iterable(CPDB_UniProtID_edgelist)))

query_string, valid_genes, invalid_genes = gct.query_constructor(CPDB_Converted_Genes)

# Set scopes (gene naming systems to search)
scopes = "uniprot"
# Set fields (systems from which to return gene names from)
fields = "symbol, entrezgene"
# Query MyGene.Info
match_list = gct.query_batch(query_string, scopes=scopes, fields=fields)
print len(match_list), 'Matched query results'

match_table_trim, query_to_symbol, query_to_entrez = gct.construct_query_map_table(match_list, valid_genes)

# Convert UniProt Accession ID CPDB edgelist to gene symbols
CPDB_edgelist_symbol = gct.convert_edgelist(CPDB_UniProtID_edgelist_filt, query_to_symbol)
CPDB_edgelist_symbol_filt = gct.filter_converted_edgelist(CPDB_edgelist_symbol)

# Save CPDB as gene symbol network
gct.write_edgelist(CPDB_edgelist_symbol_filt, wd+'Network_SIFs_Symbol/ConsensusPathDB_Symbol.sif')



