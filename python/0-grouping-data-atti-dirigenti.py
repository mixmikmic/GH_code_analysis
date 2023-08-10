import json
import pandas as pd
import glob
from pandas.io.json import json_normalize
pd.options.display.max_rows = 200
pd.options.display.max_columns = 999

path = '../data/atti_dirigenti/regione-toscana*'

files = glob.glob(path)
files[:5]

columns = ['MT.CODICE_PRATICA', 'MT.DATA_ATTO', 'MT.DESCRIZIONE_TIPO_PRATICA', 'MT.ID_TIPO', 'MT.ID_TIPO_PRATICA', 
           'MT.LINK_CONTENUTO_BINARIO', 'MT.OGGETTO', 'MT.PERSONA', 'MT.STATUS', 'MT.UFFICIO_AC',
           'MT.UFFICIO_DG', 'MT.UFFICIO_SE', 'UE']

update_columns = [v[3:] for v in columns[:-1]]
update_columns.append('DETAIL_URL')

df_list = []

def unwrap_list(x):
       if type(x) == list:
           return x[0]
       else:
           return x

for file in files:
    with open(file, 'r') as r:
        data = json.loads(r.read())
        flat_records = json_normalize(data['GSP']['RES']['R'])
        df_atti = flat_records[columns]
        df_atti.columns = update_columns
        df_atti = df_atti.applymap(unwrap_list)
        df_list.append(df_atti)

len(df_list)

df_atti = pd.concat(df_list)

df_atti.shape

df_atti.head()

df_atti['STATUS'] = df_atti['STATUS'].astype('int64')
df_atti['DATA_ATTO'] = pd.to_datetime(df_atti.DATA_ATTO)

df_atti.dtypes

df_atti.to_pickle('../data/atti-dirigenti.pkl')

