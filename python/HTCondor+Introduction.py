import htcondor
import classad

coll = htcondor.Collector() # Create the object representing the collector.
schedd_ad = coll.locate(htcondor.DaemonTypes.Schedd) # Locate the default schedd.
print schedd_ad['MyAddress'] # Prints the location of the schedd, using HTCondor's internal addressing scheme.

coll.query(htcondor.AdTypes.Schedd, projection=["Name", "MyAddress", "DaemonCoreDutyCycle"])

import socket # We'll use this to automatically fill in our hostname
coll.query(htcondor.AdTypes.Schedd, constraint='Name=?=%s' % classad.quote("jovyan@%s" % socket.getfqdn()), projection=["Name", 
"MyAddress", "DaemonCoreDutyCycle"])

schedd = htcondor.Schedd()
schedd = htcondor.Schedd(schedd_ad)
print schedd

sub = htcondor.Submit()
sub['executable'] = '/bin/sleep'
sub['arguments'] = '5m'
with schedd.transaction() as txn:
    sub.queue(txn, 10)

for job in schedd.xquery(projection=['ClusterId', 'ProcId', 'JobStatus']):
    print job.__repr__()

for job in schedd.xquery(requirements = 'ProcId >= 5', projection=['ProcId']):
    print job.get('ProcId')

print coll.query(htcondor.AdTypes.Startd, projection=['Name', 'Status', 'Activity', 'JobId', 'RemoteOwner'])[0]

