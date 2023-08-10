from qumulo.rest_client import RestClient

# change these to your Qumulo API cluster and credentials
rc = RestClient('<qumulo-cluster>', 8000)
rc.login('<login>', '<password>');

activity_data = rc.analytics.current_activity_get()

list(set([d['type'] for d in activity_data['entries']]))

types_data = {}
for d in activity_data['entries']:
    if d['type'] not in types_data:
        types_data[d['type']] = 0
    types_data[d['type']] += d['rate']
for t, val in types_data.iteritems():
    print("%22s - %.1f" % (t, val))

ip_data = {}
for d in activity_data['entries']:
    if 'throughput' in d['type']:
        if d['ip'] not in ip_data:
            ip_data[d['ip']] = 0
        ip_data[d['ip']] += d['rate']
for ip, val in sorted(ip_data.items(), key=lambda x: x[1], reverse=True):
    print("%22s - %.1f" % (ip, val))

batch_size = 1000
ids_to_paths = {}

ids = list(set([d['id'] for d in activity_data['entries']]))

for offset in range(0, len(ids), batch_size):
    resolve_data = rc.fs.resolve_paths(ids[offset:offset+batch_size])
    for id_path in resolve_data:
        ids_to_paths[id_path['id']] = id_path['path']

print("Resolved %s ids to paths." % len(ids_to_paths))

directory_levels = 1
path_data = {}

for d in activity_data['entries']:
    path = ids_to_paths[d['id']]
    if path == '':
        path = '/'
    path = '/'.join(path.split('/')[0:directory_levels+1])
    if 'throughput' in d['type']:
        if path not in path_data:
            path_data[path] = 0
        path_data[path] += d['rate']

for path, val in sorted(path_data.items(), key=lambda x: x[1], reverse=True):
    print("%32s - %.1f" % (path, val))



