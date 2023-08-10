import getpass
from transmart_api import TransmartApi

api = TransmartApi(host = 'http://localhost:8080', user = 'admin', password = getpass.getpass())

observations = api.get_observations(study = 'GSE8581')
observations[0:5]

import pandas
from pandas.io.json import json_normalize
df = json_normalize(observations)
df

dfp = df.pivot(index = 'subject.inTrialId', columns = 'label', values = 'value')
dfp

get_ipython().magic('load_ext rpy2.ipython')

get_ipython().run_cell_magic('R', '-i dfp', '\nplot(c(dfp$X.Public.Studies.GSE8581.Subjects.Age), c(dfp$X.Public.Studies.GSE8581.Subjects.Height..inch..))\n\nstr(dfp)\n')

(hdHeader, hdRows) = api.get_hd_node_data(study = 'GSE8581', node_name = 'Lung')

#1 - double
#2 - string
[(x.name, x.type) for x in hdHeader.columnSpec]

hdDataDic = {row.label: row.value[1].doubleValue for row in hdRows}

from pandas import DataFrame
hdDataDic['patientId'] = [assay.patientId for assay in hdHeader.assay]
assayIds = [assay.assayId for assay in hdHeader.assay]
DataFrame(data=hdDataDic, index = assayIds)

