import urllib2
import json
import requests
import logging
import pandas
from collections import Counter
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(message)s',)

#Some helper query functions to API
base_url = 'http://isbtranslatorapi.adversary.us'
def query_isb(endpoint, data={}, base_url=base_url):
    req = requests.post('%s/%s' % (base_url,endpoint), data=data)
    return req.json()

def get_analytes(kwargs):
    kw_local = kwargs.copy()
    frm = 0
    size = 1000
    meta = []
    kw_local['from'] = frm
    kw_local['size'] = size
    res = query_isb('/v1/analyte', data=kw_local)
    meta += res
    # Note: this is relying on the pagination, it would be smarter to just partition
    # the *sig_ids* set which would greatly speed up the query
    while len(res) > 0:
        kw_local['from'] += size
        logging.debug("Saving records from %i to %i" %(frm, frm+size))
        res = query_isb('/v1/analyte', data=kw_local)
        meta+=res
    return meta

def get_correlations(kwargs):
    kw_local = kwargs.copy()
    sigs = []
    frm = 0
    size = 10000
    meta = []
    kw_local['from'] = frm
    kw_local['size'] = size
    res = query_isb('v1/correlation', data=kw_local)
    correlations = res[:]
    while len(res) > 0:
        logging.debug("Saving records from %i to %i" %(frm, frm+size))
        kw_local['from'] += size
        frm = kw_local['from']
        res = query_isb('v1/correlation', data=kw_local)
        correlations += res
    return correlations

drugs = pandas.read_csv('patientdrugs.csv')
drugs.columns=['OHDSI_id',  "count","drug_name"]
drugs = drugs.drop(0)
drugs

drugs[drugs.OHDSI_id == 19019113]

import scipy.stats
control_drugs = pandas.read_csv('controldrugs.csv')

control_drugs

control_drugs.columns=['OHDSI_id',  "control_count","drug_name"]

control_drugs=control_drugs.set_index('OHDSI_id')

control_drugs

drugs['control_count'] = control_drugs.loc[drugs.OHDSI_id.tolist()]['control_count'].tolist()
drugs = drugs.fillna(0)
drugs = drugs[drugs['count'] > 20]
d_sum = drugs.sum()
drugs['odds_ratio'] = None
drugs['pvalue'] = None
for i, d in drugs.iterrows():
    odds, pv = scipy.stats.fisher_exact([[d['count'], d['control_count']], [d_sum['count'], d_sum['control_count']]])
    drugs.loc[i,'odds_ratio'] = odds
    drugs.loc[i, 'pvalue'] = pv
my_drugs = drugs[(drugs.pvalue < .01/len(drugs)) & (drugs.odds_ratio > 1.0)].sort_values('pvalue')
my_drugs

import re

def get_targets_from_drug_table(drugs):
    target_map = {}
    ctr = 1 
    for i,r in drugs.iterrows():
        if i > 0:
            query = 'http://c.biothings.io/v1/query?q=drugbank.name:%s' % re.sub(r'[^a-zA-Z0-9-_\s]', '', r['drug_name'])
            req = requests.get(query)
            res = req.json()
            if 'success' in res and not res['success']:
                print "Error"
                print query
                print res
            else:
                if res['total'] > 0:
                    target_map[i] = {'full_response':res}
                    for h in res['hits']:
                        if 'drugbank' in h:
                            if 'targets' in h['drugbank']:
                                if 'targets' not in target_map[i]:
                                    target_map[i]['targets'] = []
                                target_map[i]['targets'].append(h['drugbank']['targets'])
    ups = {}
    for k, v in target_map.items():
        ups[k] = []
        if 'targets' in v:
            for t in v['targets']:
                if type(t) is list:
                    for x in t:
                        if 'uniprot' in x:  
                            ups[k].append(x['uniprot'])
                else:
                    if 'uniprot' in t:
                        ups[k].append(t['uniprot'])
    return ups

from biothings_client import get_client
from biothings_explorer import IdListHandler
def p100_protein_to_uniprot_map():
    """Map the p100 proteins to uniprot ids
    """
    prot_vars = get_analytes({'category':"Proteomics"})
    no_up = []
    up_to_prot = {}
    #map the ones that actually have uniprot ids
    for v in prot_vars:
        if 'uniprot' in v:
            up = v['uniprot']
            if up not in up_to_prot:
                up_to_prot[up] = []
            up_to_prot[up].append(v)
        else:
            no_up.append(v)
    # ones without uniprot ids

    md = get_client('drug')
    ih = IdListHandler()
    missing = []
    still_missing = []
    for prot in no_up:
        req = requests.get('http://mygene.info/v3/query?q=symbol:%s' % (prot['abbreviation'],))
        res = req.json()
        if res['total'] > 0:
            egs = map(str,[x['entrezgene'] for x in res['hits'] if 'entrezgene' in x])
            uniprot_list = ih.list_handler(input_id_list=egs, input_type='entrez_gene_id', output_type='uniprot_id')
            if len(uniprot_list):
                for up in uniprot_list:
                    if up not in up_to_prot:
                        up_to_prot[up] = []
                    up_to_prot[up].append(prot)
            else:
                still_missing.append(prot)
        else:
            still_missing.append(prot)
    still_missing2 = []
    for prot in still_missing:
        req = requests.get('http://mygene.info/v3/query?q=symbol:%s' % (prot['abbreviation'].replace('_',''),))
        res = req.json()
        if res['total'] > 0:
            egs = map(str,[x['entrezgene'] for x in res['hits'] if 'entrezgene' in x])
            uniprot_list = ih.list_handler(input_id_list=egs, input_type='entrez_gene_id', output_type='uniprot_id')
            if len(uniprot_list):
                for up in uniprot_list:
                    if up not in up_to_prot:
                        up_to_prot[up] = []
                    up_to_prot[up].append(prot)
            else:
                still_missing2.append(prot)
        else:
            still_missing2.append(prot)
    print "%i unmapped proteins, %i mapped proteins" % (len(still_missing2), len(up_to_prot))
    return up_to_prot 

# lets find protein
def from_dt_drugs_to_p100_proteins(ups, up_to_prot):
    targets_to_drugs = {}
    for k, v in ups.items():
        for upd in v:
            if upd in up_to_prot:
                if k not in targets_to_drugs:
                    targets_to_drugs[k] = []
                targets_to_drugs[k].append(up_to_prot[upd])
    return targets_to_drugs

def describe_network(subnet, drug_name):
    print "%s targets %s" % (drug_name, ','.join(subnet['target']))
    print "%i edges in HPWP in %s seeded network." % (len(subnet['edges']), drug_name,)
    num_nodes = len(subnet['nodes'])
    print "%i total nodes in HPWP %s seeded subnetwork" % (num_nodes, drug_name)
    for cat, count in Counter([v['category'] for v in subnet['nodes'].values()]).items():
        print " - %i %s in HPWP %s seeded network" % (count, cat, drug_name)

def get_subnets(targets_to_drugs):
    neighbors = {}
    for k, v in targets_to_drugs.items():
        neighbors[k] = {}
        id_list = list(set([prot['_id'] for prot in v[0]]))
        neighbors[k]['target'] = id_list
        #return neighbors of target
        acorr = get_correlations({'ids1':','.join(id_list), 'bh_adjusted_pvalue':.1})
        adf = pandas.DataFrame(acorr)
        nodes = set(adf._id_1.tolist() + adf._id_2.tolist())
        my_nodes = {a['_id']: a for a in get_analytes({'ids':','.join(nodes)})}
        # get the connecting edges
        acorr = get_correlations({'ids1':','.join(my_nodes.keys()), 'ids2':','.join(my_nodes.keys())
                                  , 'bh_adjusted_pvalue':.1})
        adf = pandas.DataFrame(acorr)                 
        neighbors[k]['edges'] = adf
        neighbors[k]['nodes'] = my_nodes
    return neighbors

dt_idx_to_uniprot_target = get_targets_from_drug_table(my_drugs)

import pickle
import os
if os.path.exists('uniprot_to_p100_protein_nodes.pkl'):
    uniprot_to_p100_protein_nodes = pickle.load(open('uniprot_to_p100_protein_nodes.pkl', 'r'))
else:
    uniprot_to_p100_protein_nodes = p100_protein_to_uniprot_map()
    pickle.dump(uniprot_to_p100_protein_nodes, open('uniprot_to_p100_protein_nodes.pkl','w'))

dt_drugs_to_p100_proteins = from_dt_drugs_to_p100_proteins(dt_idx_to_uniprot_target, uniprot_to_p100_protein_nodes)

subnets = get_subnets(dt_drugs_to_p100_proteins)

condense = set()
for k, sub in subnets.items():
    #print "Index:", k
    drug_name =  my_drugs.loc[k, 'drug_name']
    if drug_name[:20] not in condense:
        condense.add(drug_name[:20])
        describe_network(sub, drug_name)
        print "="*30

print "Out of ", len(my_drugs), " drugs examined  ", len(subnets), " had direct targets in the p100 proteins out of ", len(set(sum(dt_idx_to_uniprot_target.values(), []))), " possible identified targets"
print "%.1f percent" % ((len(subnets)/float(len(my_drugs))) *100)
print "Note there are many repeated drugs at different dosages."

print "We are measuring %.1f percent of proteome" % ((len(uniprot_to_p100_protein_nodes.keys())/20000.0) * 100)



