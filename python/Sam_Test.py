import datetime as dt
import doekbase.data_api.object as object
print dt.datetime.now()
services = { "workspace_service_url": "https://ci.kbase.us/services/ws/",  "shock_service_url": "https://ci.kbase.us/services/shock-api/", }
object_api = object.ObjectAPI(services, ref="PrototypeReferenceGenomes/kb|g.3899")
print object_api.get_typestring()
print object_api.get_id()
print object_api.get_name()
print object_api.get_info()
print dt.datetime.now()

print dt.datetime.now()
data = object_api.get_data()
print dt.datetime.now()
print data

import datetime
print datetime.datetime.now()
ws = doekbase.data_api.browse('1013')
Ath = ws["kb|g.3899"]
print Ath
#CDS_List = Ath.object.get_feature_ids(type_list=['CDS'])
Proteins = Ath.object.get_proteins()
print datetime.datetime.now()
for key, value in Proteins.items():
    print value['protein_id']+' '+value['function']+' '+value['amino_acid_sequence']
    break

