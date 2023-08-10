get_ipython().system('curl -H \'Content-Type: application/json\'  --data-binary \'{"searchString":"Fanconi Anemia"}\' http://www.ndexbio.org/v2/search/network/')

import pandas as pd
import requests
from pandas.io.json import json_normalize
from urllib.parse import urlencode

pd.set_option('max_colwidth', 3800)
pd.set_option('display.expand_frame_repr', False)

ndexBase = "http://www.ndexbio.org/v2/search/network/"
headers = {'Content-Type': 'application/json'}

json = requests.post(ndexBase, headers=headers, json={'searchString': 'Fanconi Anemia'})

df = pd.DataFrame.from_records(json.json()['networks'])
df


