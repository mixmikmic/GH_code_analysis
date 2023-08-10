import requests
import pandas as pd
import copy

solr_url = 'https://solr-dev.monarchinitiative.org/solr/golr/select'
core_set = 'https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/FA_gene_sets/FA_1_core_complex.txt'

columns = ['gene', 'interactor_id', 'interactor_symbol', 'qualifier', 'inferred_gene']
dataframe = pd.read_csv(core_set, sep='\t', names=['gene', 'symbol'])

def get_solr_results(solr, params):
    resultCount = params['rows']
    while params['start'] < resultCount:
        solr_request = requests.get(solr, params=params)
        response = solr_request.json()
        resultCount = response['response']['numFound']
        params['start'] += params['rows']
        for doc in response['response']['docs']:
            yield doc

interaction_params = {
    'wt': 'json',
    'rows': 100,
    'start': 0,
    'q': '*:*',
    'fl': 'subject, subject_label, subject_closure, \
           object, object_label, object_taxon',
    'fq': ['relation_closure: "RO:0002434"']
}

# Make new dataframe for results
interact_table = pd.DataFrame(columns=columns)


# Get interactions, both direct and inferred
for index, row in dataframe.iterrows():
    params = copy.deepcopy(interaction_params)
    params['fq'].append('subject_closure: "{0}"                         OR subject_ortholog_closure: "{0}"'
                        .format(row['gene']))
    for doc in get_solr_results(solr_url, params):
        result = {}
        result['gene'] = row['symbol']
        result['interactor_id'] = doc['object']
        result['interactor_symbol'] = doc['object_label']
        if row['gene'] in doc['subject_closure']:
            result['qualifier'] = "direct"
        else:
            result['qualifier'] = "homology"    
        interact_table = interact_table.append(result, ignore_index=True)
            
interact_table.head(10)

# Define function to fetch orthologs given a gene ID
def get_human_ortholog(solr, gene):
    params = {
            'wt': 'json',
            'rows': 100,
            'start': 0,
            'q': '*:*',
            'fl': 'subject, subject_label,'
                  'object, object_label',
            'fq': ['subject_closure: "{0}"'.format(gene),
                   'relation_closure: "RO:HOM0000017"',
                   'object_taxon: "NCBITaxon:9606"'
            ]
    }
    for doc in get_solr_results(solr, params):
        yield doc

# Get interactions, both direct and inferred
for index, row in interact_table.iterrows():
    if row['qualifier'] == 'homology':
        for doc in get_human_ortholog(solr_url, row['interactor_id']):
            result = {}
            result['gene'] = row['gene']
            result['interactor_id'] = doc['object']
            result['interactor_symbol'] = doc['object_label']
            result['qualifier'] = "homology"    
            result['inferred_gene'] = row['interactor_symbol']
            interact_table = interact_table.append(result, ignore_index=True)
        
interact_table.tail(10)

# Across the list of gene pairs, which genes show up the most?

df = interact_table['interactor_symbol'].value_counts()
df.head(30)

# Filter out genes from FA set
fa_all = 'https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/FA_gene_sets/FA_4_all_genes.txt'

all_genes = pd.read_csv(fa_all, sep='\t', names=['gene', 'symbol'])

filtered_frame = interact_table[~interact_table['interactor_id'].isin(all_genes['gene'].tolist())]
filtered_frame = filtered_frame[~filtered_frame['interactor_symbol'].isin(interact_table['inferred_gene'].tolist())]


df = filtered_frame['interactor_symbol'].value_counts()
df.head(40)



