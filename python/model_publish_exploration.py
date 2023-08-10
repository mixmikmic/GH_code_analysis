import swat
import pandas as pd

cashost='localhost'
casport=5570
deploysess = swat.CAS(hostname=cashost, port=casport,  authinfo='/home/centos/.authinfo', caslib="public", name="brad")
deploysess

# http://bbviya3.pre-sal.sashq-r.openstack.sas.com:8888/view/gb_model_astore.sashdat?token=94f6909cc1e6aef5dba48190e2395583bf8e73ca4009cc45
modeltbl = deploysess.CASTable(name='gb_model_astore', caslib="PUBLIC")
if not modeltbl.tableexists().exists:
    modeltbl = deploysess.upload_file("gb_model_astore.bin.sashdat", casout=modeltbl)
#castbl = deploysess.table.loadTable(path="gb_model_astore.sashdat", casout={"name":"gb_model_astore","caslib":"public"})

# gb_model_astore = deploysess.CASTable(castbl.tableName)
# gb_model_astore.head()

modeltbl["_state_"]

deploysess.loadactionset('astore')
deploysess.loadactionset('decisiontree')
m = deploysess.describe(
     rstore=dict(name='gb_model_astore',caslib='PUBLIC') 
    )  
m.OutputVariables

import json
import pandas as pd
# j = json.load(dict(m))

df = pd.DataFrame(m.InputVariables)
df
# m.InputVariables.to_json()
for key, value in m.items(): 
    print(key)
    print(value.to_json())

deploysess.table.fileinfo(caslib="public",path="%")

