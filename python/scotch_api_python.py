import pandas as pd
import pickle
import requests
import json

features_uri = 'https://ibm.box.com/shared/static/2vntdqbozf9lzmukkeoq1lfi2pcb00j1.dataframe' 
sim_uri = 'https://ibm.box.com/shared/static/54kzs5zquv0vjycemjckjbh0n00e7m5t.dataframe'

resp = requests.get(features_uri)
resp.raise_for_status()
features_df = pickle.loads(resp.content)

resp = requests.get(sim_uri)
resp.raise_for_status()
sim_df = pickle.loads(resp.content)

features_df = features_df.drop('cluster', axis=1)

REQUEST = json.dumps({
    'path' : {},
    'args' : {}
})

# GET /scotches
names = sim_df.columns.tolist()
print(json.dumps(dict(names=names)))

# GET /scotches/:scotch
request = json.loads(REQUEST)
name = request['path'].get('scotch', 'Talisker')
features = features_df.loc[name]
# can't use to_dict because it retains numpy types which blow up when we json.dumps
print('{"features":%s}' % features.to_json())

# GET /scotches/:scotch/similar
request = json.loads(REQUEST)
name = request['path'].get('scotch', 'Talisker')
count = request['args'].get('count', 5)
inc_features = request['args'].get('include_features', True)

similar = sim_df[name].order(ascending=False)
similar.name = 'Similarity'
df = pd.DataFrame(similar).ix[1:count+1]

if inc_features:
    df = df.join(features_df)
    
df = df.reset_index().rename(columns={'Distillery': 'Name'})
result = {
    'recommendations' : [row[1].to_dict() for row in df.iterrows()],
    'for': name
}
print(json.dumps(result))

