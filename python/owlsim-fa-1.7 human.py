# autogenerate biolink_client
# curl --insecure -X POST -H "content-type:application/json" -d '{"swaggerUrl":"https://api.monarchinitiative.org/api/swagger.json"}' https://generator.swagger.io/api/gen/clients/python
# and rename it to biolink_client

import os, sys
# change this path
sys.path.insert(0, "/home/gstupp/projects/NCATS-Tangerine/biolink_client")

import biolink_client
from biolink_client.api_client import ApiClient
from biolink_client.rest import ApiException
import requests
from itertools import chain
import pandas as pd
from pprint import pprint
from tqdm import tqdm, tqdm_notebook
from collections import defaultdict

pd.options.display.max_rows = 999
pd.options.display.max_columns = 12
pd.set_option('display.width', 1000)

MONARCH_API = "https://api.monarchinitiative.org/api"
OWLSIM_API = "http://owlsim3.monarchinitiative.org/api"

gene_list = "https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/FA_gene_sets/FA_4_all_genes.txt"

client = ApiClient(host=MONARCH_API)
client.set_default_header('Content-Type', 'text/plain')
api_instance = biolink_client.BioentityApi(client)

# Get the gene list from github
dataframe = pd.read_csv(gene_list, sep='\t', names=['gene_id', 'symbol'])
df = dataframe.set_index('symbol')
human_genes = set(df.gene_id)
symbol_id = dict(zip(df.index, df.gene_id))
id_symbol = {v:k for k,v in symbol_id.items()}

gene_hpo_map = dict()
for gene_id in tqdm_notebook(set(df.gene_id)):
    api_response = api_instance.get_gene_phenotype_associations(gene_id, rows=500)
    # TODO add facet_counts to AssociationResults model
    # TODO use facet_counts to check the gene does not have >500 phenotypes
    # TODO or better, add pagination
    gene_hpo_map[gene_id] = api_response.objects

# Get the first five phenotypes for FANCA
pprint(gene_hpo_map[df.at['FANCA', 'gene_id']][0:5])

# Search for top human genes
# TODO implement prefix or taxon+type filters in owlsim
# TODO fix cutoff filter

# Note that this notebook takes a few minutes to run

# Use phenodigm algorithm
matcher = 'phenodigm'
results = []

for ncbi_id, phenotypes in tqdm_notebook(gene_hpo_map.items()):
    params = { 'id': phenotypes }
    url = "{}/match/{}".format(OWLSIM_API, matcher)
    req = requests.get(url, params=params)
    owlsim_results = req.json()
    if "matches" not in owlsim_results:
        print(ncbi_id, owlsim_results)
        continue
    for match in owlsim_results['matches']:
        results.append([ncbi_id, id_symbol[ncbi_id], match['matchId'], match['matchLabel'], match['rawScore']])

results[0]

# Create a table of query gene, matched gene, and sim score
column_names = ['query_gene', 'query_symbol', 'match_gene', 'match_symbol', 'sim_score']
df = pd.DataFrame(data=results, columns=column_names)
df = df.replace('NaN', pd.np.NaN).dropna().reindex()

# Get sim scores for ERCC4
df_ercc4 = df.query("query_symbol == 'ERCC4'")
print(df_ercc4.head(40))

# Filter out Non-Genes
df = df[df.match_gene.str.startswith("NCBIGene")]

# Get sim scores for ERCC4
df_ercc4 = df.query("query_symbol == 'ERCC4'")
print(df_ercc4.head(40))

# remove self matches
df = df[df.query_gene != df.match_gene]

# sum scores for each matched gene
sim_score = df.groupby("match_symbol").agg({"sim_score": sum}).sim_score
sim_score = sim_score.sort_values(ascending=False)
sim_score[:20]

## Sanity check. Only show the FA genes
sim_score[sim_score.index.isin(symbol_id)]

# Filter out all genes from the input set (FA)
sim_score_nofa = sim_score[~sim_score.index.isin(symbol_id)]
sim_score_nofa[:20]

# which genes matched to ERCC5?
df.query("match_symbol == 'ERCC5'")

# Across the list of gene pairs, which genes show up the most?
df['match_symbol'].value_counts()[:100]

## Run same summation, but removing all scores lower than 70 beforehand
sim_score = df.query("sim_score>70").groupby("match_symbol").agg({"sim_score": sum}).sim_score
sim_score = sim_score.sort_values(ascending=False)
sim_score = sim_score[~sim_score.index.isin(symbol_id)]
sim_score[:20]

