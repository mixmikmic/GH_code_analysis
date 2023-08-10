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

MONARCH_API = "https://api.monarchinitiative.org/api"
SIMSEARCH_API = "https://monarchinitiative.org/simsearch/phenotype"

gene_list = "https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/FA_gene_sets/FA_4_all_genes.txt"

client = ApiClient(host=MONARCH_API)
client.set_default_header('Content-Type', 'text/plain')
api_instance = biolink_client.BioentityApi(client)

# Get the gene list from github
dataframe = pd.read_csv(gene_list, sep='\t', names=['gene_id', 'symbol'])
df = dataframe.set_index('symbol')
human_genes = set(df.gene_id)

taxids = [10090, 7955, 7227, 6239]
prefixes = ['MGI', 'ZFIN', 'WormBase', 'FlyBase']

def get_obj(obj_id):
    url = "https://api.monarchinitiative.org/api/bioentity/{}".format(obj_id)
    res = requests.get(url)
    d = res.json()
    return d
def get_taxon_from_gene(gene):
    return get_obj(gene)['taxon']['label']
get_taxon_from_gene('NCBIGene:2176')

def query_orthologs(gene_id, taxon=None):
    """Query Monarch to determine the orthologs of a gene."""
    url = "https://api.monarchinitiative.org/api/bioentity/gene/{}/homologs/".format(gene_id)
    if taxon:
        res = requests.get(url, params={'homolog_taxon': taxon})
    else:        
        res = requests.get(url)
    d = res.json()
    return [x['object']['id'] for x in d['associations']]
#query_orthologs('MGI:88276', taxon="NCBITaxon:9606")

def get_phenotype_from_gene(gene):
    # https://monarchinitiative.org/gene/NCBIGene%3A2176/phenotype_list.json
    url = "https://monarchinitiative.org/gene/{}/phenotype_list.json"
    return [x['id'] for x in requests.get(url.format(gene)).json()['phenotype_list']]
def get_phenotype_from_gene_verbose(gene):
    # https://monarchinitiative.org/gene/NCBIGene%3A2176/phenotype_list.json
    url = "https://monarchinitiative.org/gene/{}/phenotype_list.json"
    return [(x['id'],x['label']) for x in requests.get(url.format(gene)).json()['phenotype_list']]
#get_phenotype_from_gene("NCBIGene:2176")

def get_phenotypically_similar_genes(phenotypes, taxon, return_all=False):
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.8',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
    }
    data = {'input_items': " ".join(phenotypes), "target_species": taxon}
    r = requests.post(SIMSEARCH_API, data=data, headers=headers)
    d = r.json()
    if return_all:
        return d
    if 'b' in d:
        scores = [(x['id'],x['score']['score'], x['label']) for x in d['b']]
    else:
        scores = []
    return scores
#get_phenotypically_similar_genes(phenotypes, "10090")

# human_genes = ["NCBIGene:2176"]
gene_genes = defaultdict(list)
for taxid in tqdm_notebook(taxids):
    for gene in tqdm_notebook(human_genes, leave=False):
        phenotypes = get_phenotype_from_gene(gene)
        gene_genes[gene].extend(get_phenotypically_similar_genes(phenotypes, taxid))

s = defaultdict(int)
gene_label = dict()
for human_gene, ortho_genes in gene_genes.items():
    for ortho_gene, score, label in ortho_genes:
        gene_label[ortho_gene] = label
        s[ortho_gene] += score

top10 = dict()
for prefix in prefixes:
    ss = {k:v for k,v in s.items() if k.startswith(prefix)}
    top10[prefix] = sorted(ss.items(), key=lambda x:x[1], reverse=True)[:20]
ss = list(chain(*top10.values()))
ss = [{'gene': s[0], 'score': s[1]} for s in ss]
ss

for s in tqdm_notebook(ss):
    s['orthologs'] = query_orthologs(s['gene'], "NCBITaxon:9606")
ss

for s in tqdm_notebook(ss):
    s['label'] = get_obj(s['gene'])['label']
    s['ortholog_labels'] = [get_obj(x)['label'] for x in s['orthologs']]

ss = sorted(ss, key=lambda x: x['score'], reverse=True)
print("\n".join([",".join([x['orthologs'][0],x['ortholog_labels'][0], str(x['score'])]) for x in ss[:20]]))

## FANCC
phenotypes = get_phenotype_from_gene_verbose("NCBIGene:7042")
phenotypes

d = get_phenotypically_similar_genes([x[0] for x in phenotypes], "10090", return_all=True)
genes = get_phenotypically_similar_genes([x[0] for x in phenotypes], "10090", return_all=False)
genes

match = d['b'][0]
(match['id'],match['label'])

match['matches'][:2]

# FANCC and Gli3 are "phenotypically similar" because of these phenotypes in common
[(x['lcs']['id'],x['lcs']['label']) for x in match['matches']]

human_orthologs = query_orthologs(match['id'], taxon="NCBITaxon:9606")
human_orthologs

for human_gene, pgenes in gene_genes.items():
    pgenes = [x for x in pgenes if "MGI:98726" == x[0]]
    print(human_gene, get_obj(human_gene)['label'], pgenes)

## Version 2 : Get orthologs first
phenotypes = get_phenotype_from_gene("MGI:88276")
get_phenotypically_similar_genes(phenotypes, "9606")

