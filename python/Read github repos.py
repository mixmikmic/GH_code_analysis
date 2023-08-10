import re
import sys
import pandas as pd
import numpy as np
from github import Github
from getpass import getpass
from pandas.io.json import json_normalize
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import pickle
get_ipython().magic('matplotlib inline')

repos = pd.read_csv("awe_repos_100_in_cats.csv", encoding="utf-8")

repos.shape

ACCESS_TOKEN = getpass(prompt="Enter github token:")
client = Github(ACCESS_TOKEN, per_page=100)

cache = {
    'repos':{},
    'contributors':{},
    'followers':{}
}

get_ipython().run_cell_magic('time', '', '\nSTART_FROM = 0 #index\nfor i, repo_name in enumerate(repos.iloc[START_FROM:]["repo"].values):\n    if ((START_FROM+i) % 100) == 0: \n        print(START_FROM+i,repo_name)\n    try:\n        repo = client.get_repo(repo_name)\n        repo.raw_data\n\n        cache[\'repos\'][repo_name] = repo\n        \n        \n        \n    except Exception, e: #ssl.SSLError\n        print >> sys.stderr, "Encountered an error fetching data for", \\\n                             repo_name, "Skipping."\n        print >> sys.stderr, e\n\n    #if ((START_FROM+i) % 100) == 0: print "Processed", START_FROM+i+1, " ", repo_name," nodes. Num nodes/edges in graph", \\\n    #      g.number_of_nodes(), "/", g.number_of_edges()\n    if client.rate_limiting < 400: print("Rate limit remaining", client.rate_limiting)\n    \nnx.write_gpickle(g, "github.BAK.gpickle")    \npickle.dump( cache, open( "cache.github.BAK.pickle", "wb" ) )')

repo = client.get_repo("j0k3r/tourdefrance-slack-bot")
repo.raw_data

ie = repo.get_issues_events()

for e in ie[:2]:
    print(e.raw_data)

cols = repo.get_stats_contributors()

cols

for c in cols[:2]:
    print(c.raw_data)

nx.write_gpickle(g, "github.LONG5.gpickle")

mtsw_users = [n for n in g if g.node[n]['type'] == 'user']
h = g.subgraph(mtsw_users)

#nx.write_gexf(h, "github2.gexf")
#nx.write_gml(h, "github2.gml")
nx.write_graphml(h, "github1770.graphml")

res = []
for name, _repo in cache["repos"].iteritems():
    df = json_normalize(_repo.raw_data)
    res.append(df)
repos_df = pd.concat(res, ignore_index=True)

cache["repos"].iteritems().next()

repos_df.to_csv("repos8087.csv", encoding="utf-8")

