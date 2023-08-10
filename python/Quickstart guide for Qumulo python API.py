import os
from qumulo.rest_client import RestClient

# set your environment variables or fill in the variables below
API_HOSTNAME = os.environ['API_HOSTNAME'] if 'API_HOSTNAME' in os.environ else '{your-cluster-hostname}'
API_USER =     os.environ['API_USER']     if 'API_USER'     in os.environ else '{api-cluster-user}'
API_PASSWORD = os.environ['API_PASSWORD'] if 'API_PASSWORD' in os.environ else '{api-cluster-password}'

rc = RestClient(API_HOSTNAME, 8000)
rc.login(API_USER, API_PASSWORD)
rc.auth.who_am_i()

all_nodes = rc.cluster.list_nodes()
for node in all_nodes:
    print "Node: %(id)2s/%(node_name)s - Status: %(node_status)s - Serial: %(serial_number)s" % node

file_system_stats = rc.fs.read_fs_stats()
free_bytes = float(file_system_stats['free_size_bytes'])
usable_bytes = float(file_system_stats['total_size_bytes'])
used_perc = (usable_bytes-free_bytes) / usable_bytes
print("Usable Capacity (TB): %s" % (round(usable_bytes/pow(1000,4), 2), ))
print("Used Capacity   (TB): %s (%s%%)" % (round((usable_bytes-free_bytes)/pow(1000,4), 2), round(100*used_perc,1)))

protection_details = rc.cluster.get_protection_status()
data_str = """
                    Type: %(protection_system_type)s
           Encoding size: %(blocks_per_stripe)s/%(data_blocks_per_stripe)s
      Max drive failures: %(max_drive_failures)s
       Max node failures: %(max_node_failures)s
Remaining drive failures: %(remaining_drive_failures)s
 Remaining node failures: %(remaining_node_failures)s
"""
print(data_str % protection_details)

all_drives = rc.cluster.get_cluster_slots_status()
node_num = 0
for drive in all_drives:
    if drive['node_id'] != node_num:
        print "-" * 60
        node_num = drive['node_id']
    cap = float(drive['capacity'])/pow(1000,4)
    if cap > 1:
        cap = str(int(cap)) + " TB"
    else:
        cap = str(int(cap * 1001)) + " GB"
    drive['cap'] = cap
    print "%(node_id)4s.%(slot)s - %(slot_type)s - %(state)s - %(cap)6s - %(disk_model)s" % drive



