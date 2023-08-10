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

clin_vars = get_analytes({'category':"Metabolites"})
thyroxine = [x['_id'] for x in clin_vars if x['_id'].find('thyroxine') >-1]
for met in thyroxine:
    print met

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
# get correlation network based on seed network
acorr = get_correlations({'ids1':','.join(thyroxine)})
adf = pandas.DataFrame(acorr)
sig_adf = adf[adf.bh_adjusted_pvalue < .1]
nodes = set(sig_adf._id_1.tolist() + sig_adf._id_2.tolist())
my_nodes = {a['_id']: a for a in get_analytes({'ids':','.join(nodes)})}
num_edges = len(sig_adf)
print "%i edges in HPWP in thyroxine seeded network." % (num_edges,)
num_nodes = len(my_nodes)
print "%i total nodes in HPWP thyroxine seeded subnetwork" % (num_nodes,)
for k, v in Counter([v['category'] for v in my_nodes.values()]).items():
    print "%i %s in HPWP Adenosine seeded network" % (v, k)

rec = sig_adf.to_dict('records')
json.dump(rec, open('thyroxine_sig_edges.json', 'w'))

import scipy.stats
super_pathways_condition = []
for k, v in my_nodes.items():
    if 'super_pathway' in v:
        super_pathways_condition.append(v['super_pathway'])
spc = Counter(super_pathways_condition)
super_pathways_background = []
for v in clin_vars:
    super_pathways_background.append(v['super_pathway'])
spb = Counter(super_pathways_background)
df = pandas.DataFrame([spc, spb], index=['Thyroxine-seeded HPWP', 'Total HPWP']).fillna(0.0)
perc = (df.iloc[0]/df.iloc[1])*100
perc.name = "Percent"
df = df.append(perc)
s = df.sum(axis=1)
not_ashpwp_total = s['Total HPWP'] - s['Thyroxine-seeded HPWP']
odr = {}
pv = {}
for c in df.columns:
    ct = [[df.loc['Thyroxine-seeded HPWP',c],  df.loc['Total HPWP',c]-df.loc['Thyroxine-seeded HPWP',c]],
          [s['Thyroxine-seeded HPWP'], not_ashpwp_total]
    ]
    res = scipy.stats.fisher_exact(ct)
    odr[c] = res[0]
    pv[c] = res[1]
df = df.append(pandas.Series(odr, name="Fisher OR"))
df = df.append(pandas.Series(pv, name="Fisher p-value"))
df.transpose().sort_values("Fisher p-value")

import scipy.stats
super_pathways_condition = []
for k, v in my_nodes.items():
    if 'sub_pathway' in v:
        super_pathways_condition.append(v['sub_pathway'])
spc = Counter(super_pathways_condition)
super_pathways_background = []
for v in clin_vars:
    super_pathways_background.append(v['sub_pathway'])
spb = Counter(super_pathways_background)
df = pandas.DataFrame([spc, spb], index=['Thyroxine-seeded HPWP', 'Total HPWP']).fillna(0)
df = df.fillna(0.0)
perc = (df.iloc[0]/df.iloc[1])*100
perc.name = "Percent"
df = df.append(perc)
s = df.sum(axis=1)
not_ashpwp_total = s['Total HPWP'] - s['Thyroxine-seeded HPWP']
odr = {}
pv = {}
for c in df.columns:
    ct = [[df.loc['Thyroxine-seeded HPWP',c],  df.loc['Total HPWP',c]-df.loc['Thyroxine-seeded HPWP',c]],
          [s['Thyroxine-seeded HPWP'], not_ashpwp_total]
    ]
    res = scipy.stats.fisher_exact(ct)
    odr[c] = res[0]
    pv[c] = res[1]
df = df.append(pandas.Series(odr, name="Fisher OR"))
df = df.append(pandas.Series(pv, name="Fisher p-value"))
df.transpose().sort_values('Fisher p-value')

print "Clinical Labs"
cl = pandas.DataFrame([v for v in my_nodes.values() if v['category'] == "Clinical Labs"])
sig_adf[sig_adf._id_1.isin(cl._id.tolist()) | sig_adf._id_2.isin(cl._id.tolist())]



