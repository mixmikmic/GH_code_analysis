import htcondor

sub = htcondor.Submit({"foo": "1", "bar": "2", "baz": "$(foo)"})
sub.setdefault("qux", "3")
print "=== START ===\n%s\n=== END ===" % sub
print sub.expand("baz")

sub = htcondor.Submit({"executable": "/bin/sleep", "arguments": "5m"})

schedd = htcondor.Schedd()         # Create a schedd object using default settings.
with schedd.transaction() as txn:  # txn will now represent the transaction.
   print sub.queue(txn)            # Queues one job in the current transaction; returns job's cluster ID

with schedd.transaction() as txn:
    clusterId = sub.queue(txn, 5)  # Queues 5 copies of this job.
    schedd.edit(["%d.0" % clusterId, "%d.1" % clusterId], "foo", '"bar"') # Sets attribute foo to the string "bar".
print "=== START JOB STATUS ==="
for job in schedd.xquery(requirements="ClusterId == %d" % clusterId, projection=["ProcId", "foo", "JobStatus"]):
    print "%d: foo=%s, job_status = %d" % (job.get("ProcId"), job.get("foo", "default_string"), job["JobStatus"])
print "=== END ==="

schedd.act(htcondor.JobAction.Hold, 'ClusterId==%d && ProcId >= 2' % clusterId)
print "=== START JOB STATUS ==="
for job in schedd.xquery(requirements="ClusterId == %d" % clusterId, projection=["ProcId", "foo", "JobStatus"]):
    print "%d: foo=%s, job_status = %d" % (job.get("ProcId"), job.get("foo", "default_string"), job["JobStatus"])
print "=== END ==="

