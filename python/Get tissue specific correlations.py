import urllib2
import json
import requests
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',)

base_url = 'http://isbtranslatorapi.adversary.us'

# The base metadata takes a minute.  It is counting a lot of stuff
def query_isb(endpoint, data={}, base_url=base_url):
    req = requests.post('%s/%s' % (base_url,endpoint), data=data)
    return req.json()

def get_analytes(kwargs):
    kw_local = kwargs.copy()
    frm = 0
    size = 10000
    meta = []
    kw_local['from'] = frm
    kw_local['size'] = size
    res = query_isb('/v1/analyte', data=kw_local)
    meta += res
    # Note: this is relying on the pagination, it would be smarter to just partition
    # the *sig_ids* set which would greatly speed up the query
    while len(res) > 0:
        kw_local['from'] += size
        frm = kw_local['from']
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

def get_correlations_undumb(kwargs, idlist=[]):
    kw_local = kwargs.copy()
    sigs = []
    frm = 0
    size = 50
    corr = []
    sub_ids = idlist[frm:frm+size]
    kw_local['ids1'] = ','.join(sub_ids)
    kw_local['size'] = size*size 
    prev = 0
    corr += query_isb('v1/correlation', data=kw_local)
    while len(sub_ids) > 0:
        frm += size
        sub_ids = idlist[frm:frm+size]
        kw_local = kwargs.copy()
        kw_local['ids1'] = ','.join(sub_ids)
        kw_local['size'] = size*size 
        corr += query_isb('v1/correlation', data=kw_local)
        curr = len(corr) - prev
        logging.debug("%i records"  % (len(corr,)))
        prev = curr

print "You can view see what values are associated with a field"
print "Note most queries are paginated using from and size."

print json.dumps( query_isb('v1/metadata/analytes', data={'from':0, 'size':50}), indent=2)

print "You can view see what values are associated with a field"
print "Note most queries are paginated using from and size."

print json.dumps( query_isb('v1/metadata/analytes/tissue', data={'from':0, 'size':33}), indent=2)



corr_nets = {}
fanc_genes = [x.strip() for x in 'BRCA1, BRCA2, BRIP1, ERCC4, FANCA, FANCB, FANCC, FANCD2, FANCE, FANCF, FANCG, FANCI, FANCL, FANCM, PALB2, RAD51, RAD51C, UBE2T, XRCC2'.split(',')]
tissues = ["Skin", "Kidney","Heart"]
genes = {}
for tis in tissues:
    kwargs = {'category':"gene expression", 'tissue':tis, 'source_project':'GTEx'}
    t_anal = get_analytes(kwargs)  
    genes[tis] = t_anal
    anal_ids = [t['_id'] for t in t_anal if t['abbreviation'] in fanc_genes]
    corr = get_correlations({'ids1': ','.join(anal_ids)})
    corr_nets[tis] = pandas.DataFrame(corr)

len(corr_nets['Heart'][corr_nets['Heart'].pvalue < .05/len(corr_nets['Heart'].pvalue)])



