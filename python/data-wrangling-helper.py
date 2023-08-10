import json
import pandas as pd
import glob
from pandas.io.json import json_normalize
pd.options.display.max_rows = 200
pd.options.display.max_columns = 999

dirigenti_path = 'regione-toscana/atti_dirigenti/dirigenti.json'
out_dirigenti_path = 'regione-toscana/atti_dirigenti/dirigenti_processed.json'

with open(dirigenti_path, 'r') as f:
    dirigenti = json.loads(f.read())
    dirigenti = dirigenti['getDirigenti']['Dirigente']    

with open(out_dirigenti_path, 'w') as f:
    json.dump(dirigenti, f)

df_dirigenti= pd.read_json(out_dirigenti_path)
df_dirigenti.head()

office_path = 'regione-toscana/atti_dirigenti/strutture.json'
out_office_path = 'regione-toscana/atti_dirigenti/strutture_processed.json'

with open(office_path, 'r') as f:
    offices = json.loads(f.read())
    offices = offices['getDg']['Struttura']
    
with open(out_office_path, 'w') as f:
    json.dump(offices, f)
    
df_offices = pd.read_json(out_office_path)
df_offices.head()

path = 'regione-toscana/atti_dirigenti/regione-toscana*'

files = glob.glob(path)
files[:5]

with open(files[0], 'r') as r:
    data = json.loads(r.read())
    records = data['GSP']['RES']['R']

records[0]

columns = ['MT.CODICE_PRATICA', 'MT.DATA_ATTO', 'MT.DESCRIZIONE_TIPO_PRATICA', 'MT.ID_TIPO', 'MT.ID_TIPO_PRATICA', 
           'MT.LINK_CONTENUTO_BINARIO', 'MT.OGGETTO', 'MT.PERSONA', 'MT.STATUS', 'MT.UFFICIO_AC',
           'MT.UFFICIO_DG', 'MT.UFFICIO_SE', 'UE']

update_columns = [v[3:] for v in columns[:-1]]
update_columns.append('DETAIL_URL')

update_columns

flat_records = json_normalize(records)

df_atti = flat_records[columns]
df_atti.columns = update_columns

def process_values(x):
    if type(x) == list:
        return x[0]
    else:
        return x

df_atti = df_atti.applymap(process_values)

df_atti.head()



