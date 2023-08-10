# connect to a GIS
from arcgis.gis import GIS
import arcgis.features
gis = GIS() # connect to www.arcgis.com anonymously. 
            # we will use a public sync enabled feature layer

url = 'https://sampleserver6.arcgisonline.com/arcgis/rest/services/Sync/WildfireSync/FeatureServer/'
wildfire_flc = arcgis.features.FeatureLayerCollection(url, gis)

type(wildfire_flc)

wildfire_flc.layers

# query syncEnabled property to verify is sync is enabled
wildfire_flc.properties.syncEnabled

# query the syncCapabilities property to view fine grained capabilities
wildfire_flc.properties.syncCapabilities

replica_list = wildfire_flc.replicas.get_list()
len(replica_list)

replica_list[0]

# list all capabilities
wildfire_flc.properties.capabilities

portal_gis = GIS("portal url", "username", "password")
search_result = portal_gis.content.search("Ports along west coast", "Feature Layer")

search_result[0]

ports_flc = arcgis.features.FeatureLayerCollection.fromitem(search_result[0])
type(ports_flc)

ports_flc.properties.capabilities

ports_flc = arcgis.features.FeatureLayerCollection.fromitem(sr[0])

replica1 = ports_flc.replicas.create(replica_name = 'arcgis_python_api_2',
                                    layers='0',
                                    data_format='filegdb',
                                    out_path = 'E:\\demo')
replica1

# Let us query all the replicas registered on the ports feature layer from before
replica_list = ports_flc.replicas.get_list()

for r in replica_list:
    print(r)

replica1 = ports_flc.replicas.get('86E9D1D7-96FF-4B40-A366-DC9A9AAB6923')
replica1

import time
time.localtime(replica1['creationDate']/1000) #dividing by 1000 to convert micro seconds to seconds

ten_min_earlier_epoch = time.time() - 10
ten_min_earlier_epoch

import time
removal_list = []
for r in replica_list:
    temp_r = ports_flc.replicas.get(r['replicaID'])
    temp_dict = {'replica_id': r['replicaID'],
                'creationDate':temp_r['creationDate']/1000}
    
    #check
    if temp_dict['creationDate'] < ten_min_earlier_epoch:
        removal_list.append(temp_dict)
        print(temp_dict)

for r in removal_list:
    result = ports_flc.replicas.unregister(r['replica_id'])
    print(result)

