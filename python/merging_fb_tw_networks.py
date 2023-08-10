import psycopg2

import pandas as pd
import numpy as np

from copy import deepcopy

import ast

import networkx as nx

import itertools

import matplotlib.pyplot as plt

sehir_matches_by_id = pd.read_csv('datasets/twitter_fb_matches.csv', index_col="ID").set_index("twitter_ID")
sehir_matches_by_id.head(5)

len(sehir_matches_by_id)

connection = psycopg2.connect('dbname=twitter_accounts_new host=localhost user=postgres password=1_sehir_1')

twitter_users = pd.read_sql("SELECT * FROM twitter_user", connection).rename(columns={'id': 'GUID', 
                 'name': 'twitter_name',
                 'description': 'profile_description',
                 'screen_name': 'twitter_screen_name'})

user_connections = pd.read_sql("SELECT * FROM twitter_connection", connection).drop('id', axis=1)

G = nx.Graph()
users_ids = dict()  # user name > id
ids_users = dict()  # user id > (twitter_ID,username)

for i, row in user_connections.iterrows():
    from_ = row['from_user_id']
    to = row['to_user_id']
    try:
        from_name = sehir_matches_by_id.loc[from_]["sehir_matches"]
        to_name = sehir_matches_by_id.loc[to]["sehir_matches"]
    except (KeyError,TypeError) as e:
        continue
#         print(from_name, to_name)
    if "sehir" in from_name or "sehir" in to_name:
        continue
    if type(from_name) == str:
        from_name = [from_name]
    if type(to_name) == str:
        to_name = [to_name]
    from_to = itertools.product(list(from_name), list(to_name))
    for f,t in from_to:
        users_ids.setdefault(f, len(users_ids))
        ids_users[users_ids[f]] = (from_, f)
        
        users_ids.setdefault(t, len(users_ids))
        ids_users[users_ids[t]] = (to, t)
        
        G.add_edge(users_ids[f],users_ids[t])

len(users_ids),G.number_of_nodes()

G.number_of_edges()

for ix in list(G.nodes()):
    twitter_id, sehir_name = ids_users[ix]
    match = sehir_matches_by_id.loc[twitter_id]
    if type(match) == pd.DataFrame: # handling duplicate rows for the same twitter ids
        # handling duplicate rows for the same twitter ids and sehir name (different fb ids)
        if type(match.set_index("sehir_matches").loc[sehir_name]) == pd.DataFrame:
            match = list(match.iterrows())[0][1]  # just picking the first ones
    for k, v in match.items():
        try:
            G.node[ix][k] = int(v)  # making sure it is not np.int64 to appeal to JSON
        except ValueError:
            G.node[ix][k] = str(v)

for ix,deg in list(G.degree()):
    G.node[ix]['degree'] = deg
    G.node[ix]['parity'] = (1-deg%2)

evc = nx.eigenvector_centrality(G)
closeness = nx.closeness_centrality(G)
betweenness = nx.betweenness_centrality(G)

metrics = {"eigenvector_centrality":evc,
           "closeness_centrality":closeness,
          "betweenness":betweenness}

for metric_name, metric in metrics.items():
    for ix,v in metric.items():
        G.node[ix][metric_name] = v

list(G.nodes(data=True))[0]

import json
from networkx.readwrite import json_graph
data = nx.node_link_data(G)
with open('twitter_fb.json', 'w') as f:
    json.dump(data, f, indent=4)



