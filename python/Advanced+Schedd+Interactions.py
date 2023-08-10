import htcondor

schedd = htcondor.Schedd()
sub = htcondor.Submit({
                       "executable": "/bin/sleep",
                       "arguments":  "5m",
                       "hold":       "True",
                      })
with schedd.transaction() as txn:
    clusterId = sub.queue(txn, 10)
print clusterId

print sum(1 for _ in schedd.xquery(projection=["ProcID"], requirements="ClusterId==%d" % clusterId, limit=5))

jobs = []
for schedd_ad in htcondor.Collector().locateAll(htcondor.DaemonTypes.Schedd):
    schedd = htcondor.Schedd(schedd_ad)
    jobs += schedd.xquery()
print len(jobs)

queries = []
coll_query = htcondor.Collector().locateAll(htcondor.DaemonTypes.Schedd)
for schedd_ad in coll_query:
    schedd_obj = htcondor.Schedd(schedd_ad)
    queries.append(schedd_obj.xquery())

job_counts = {}
for query in htcondor.poll(queries):
   schedd_name = query.tag()
   job_counts.setdefault(schedd_name, 0)
   count = len(query.nextAdsNonBlocking())
   job_counts[schedd_name] += count
   print "Got %d results from %s." % (count, schedd_name)
print job_counts

schedd = htcondor.Schedd()
for ad in schedd.history('true', ['ProcId', 'ClusterId', 'JobStatus', 'WallDuration'], 2):
    print ad

import os.path
schedd = htcondor.Schedd()
job_ad = {      'Cmd': '/bin/sh',
     'JobUniverse': 5,
     'Iwd': os.path.abspath("/tmp"),
     'Out': 'testclaim.out',
     'Err': 'testclaim.err',
     'Arguments': 'sleep 5m',
 }
clusterId = schedd.submit(job_ad, count=2)
print clusterId

foo = {'myAttr': 'foo'}
bar = {'myAttr': 'bar'}
clusterId = schedd.submitMany(job_ad, [(foo, 2), (bar, 2)])
print list(schedd.xquery('ClusterId==%d' % clusterId, ['ProcId', 'myAttr']))

ads = []
cluster = schedd.submit(job_ad, 1, spool=True, ad_results=ads)
schedd.spool(ads)

schedd.retrieve("ClusterId == %d" % cluster)

coll = htcondor.Collector()
private_ads = coll.query(htcondor.AdTypes.StartdPrivate)
startd_ads = coll.query(htcondor.AdTypes.Startd)
claim_ads = []
for ad in startd_ads:
    if "Name" not in ad: continue
    found_private = False
    for pvt_ad in private_ads:
        if pvt_ad.get('Name') == ad['Name']:
            found_private = True
            ad['ClaimId'] = pvt_ad['Capability']
            claim_ads.append(ad)

with htcondor.Schedd().negotiate("bbockelm@unl.edu") as session:
    found_claim = False
    for resource_request in session:
        for claim_ad in claim_ads:
            if resource_request.symmetricMatch(claim_ad):
                print "Sending claim for", claim_ad["Name"]
                session.sendClaim(claim_ads[0])
                found_claim = True
                break
        if found_claim: break

