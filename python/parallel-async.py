# re-connect back to the cluster
client2 = Client(profile = 'sample')

# re-load the message identifiers
with open('jobs.pickle', 'rb') as handle:
    jobs = pickle.load(handle)
    
# query job state
status = [ client2.get_result(msg).ready() for msg in jobs ]
print '{n} jobs running, {c}% complete'.format(n = len(status),
                                               c = int(((len(filter(lambda f: f, status)) + 0.0) / len(status)) * 100))

# tidy up
client2.close()

# retrieve the results
client2 = Client(profile = 'sample')
results = client2.get_result(jobs).get()
client2.close()

