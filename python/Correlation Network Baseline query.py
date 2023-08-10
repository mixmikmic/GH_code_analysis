import urllib2
import json
import requests

base_url = 'http://isbtranslatorapi.adversary.us'

# The base metadata takes a minute.  It is counting a lot of stuff
def query_isb(endpoint, data={}, base_url=base_url):
    req = requests.post('%s/%s' % (base_url,endpoint), data=data)
    return req.json()

print json.dumps( query_isb('v1/metadata'), indent=2)

print "These are the queryable fields."
print json.dumps( query_isb('v1/metadata/analytes'), indent=2)

print "You can view see what values are associated with a field"
print "Note most queries are paginated using from and size."

print json.dumps( query_isb('v1/metadata/analytes/abbreviation', data={'from':0, 'size':20}), indent=2)

print json.dumps( query_isb('v1/analyte', data={'abbreviation':'CHPF2'}), indent=2)

print json.dumps( query_isb('v1/analyte', data={'abbreviation':'CHPF2,BRCA2'}), indent=2)

print json.dumps( query_isb('v1/analyte', data={'abbreviation':'CHPF2,BRCA2', "tissue":"Brain"}), indent=2)

my_analytes = query_isb('v1/analyte', data={'abbreviation':'CHPF2,BRCA2', "tissue":"Brain", 'source_project':'GTEx'})
print json.dumps(my_analytes, indent=2)

# join together some interesting ids and look for correlations
ids = ','.join([analyte['_id'] for analyte in my_analytes])
print json.dumps( query_isb('v1/correlation', data={'ids1':ids, 'size':50}), indent=2)

# same thing but drop the crumby pvalues (Bonf sig)
ids = ','.join([analyte['_id'] for analyte in my_analytes])
print json.dumps( query_isb('v1/correlation', data={'ids1':ids, 'pvalue': .05/18385}), indent=2)

# next page of bonf sig correlations
ids = ','.join([analyte['_id'] for analyte in my_analytes])
print json.dumps( query_isb('v1/correlation', data={'ids1':ids, 'pvalue': .05/18385, 'from':10, 'size':10}), indent=2)

# hmm abca10 looks interesting and I would like to learn more about it

print json.dumps(query_isb('v1/analyte/gtex.brain.ABCA10'), indent=2)

# in fact I would like to learn about all of the bsig genes

sigs = []

frm = 0
size=1000
res = query_isb('v1/correlation', data={'ids1':ids, 'pvalue': .01/18385, 'from':frm, 'size':size})
correlations = res[:]
while len(res) > 0:
    print "Saving records from %i to %i" %(frm, frm+size)
    sigs += [x['_id_1'] for x in res]
    sigs += [x['_id_2'] for x in res]
    frm += size
    res =  query_isb('v1/correlation', data={'ids1':ids, 'pvalue': .01/18385, 'from':frm, 'size':size})
    correlations += res

sigs = list(set(sigs))
print len(sigs)

sig_ids = ','.join(sigs)

frm = 0
size = 1000
meta = []
res = query_isb('/v1/analyte', data={'ids':sig_ids, 'from':frm, 'size':size})
meta += res
# Note: this is relying on the pagination, it would be smarter to just partition
# the *sig_ids* set which would greatly speed up the query
while len(res) > 0:
    frm += size
    print "Saving records from %i to %i" %(frm, frm+size)
    res = query_isb('/v1/analyte', data={'ids':sig_ids, 'from':frm, 'size':size})
    meta+=res
    

len(meta)

#faster less dumb way of doing the same as above
sig_list = sigs
sig_ids = ','.join(sigs)

frm = 0
size = 1000
res = []
sub_sig = sig_list[frm:frm+size]
res += query_isb('/v1/analyte', data={'ids':','.join(sub_sig), 'size':size})

while len(sub_sig) > 0:
    frm += size
    sub_sig = sig_list[frm:frm+size]
    print "Saving records from %i to %i" %(frm, frm+size)
    res += query_isb('/v1/analyte', data={'ids':','.join(sub_sig), 'size':size})

import pandas

corr_net = pandas.DataFrame(correlations)
corr_net.head()

metadata = pandas.DataFrame(res)
metadata.head()

