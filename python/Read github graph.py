import re
import sys
import pandas as pd
import numpy as np
from github import Github
from getpass import getpass
from pandas.io.json import json_normalize
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import networkx as nx
import pickle
get_ipython().magic('matplotlib inline')

repos = pd.read_csv("awe_repos_100_in_cats.csv", encoding="utf-8")

repos.shape

ACCESS_TOKEN = getpass(prompt="Enter github token:")
client = Github(ACCESS_TOKEN, per_page=100)

#g = nx.DiGraph() #or read it from disk 
g = nx.read_gpickle("github.BAK.gpickle")
cache = {
    'repos':{},
    'contributors':{},
    'followers':{}
}

mtsw_repo = [n for n in g if g.node[n]['type'] == 'repo']
len(mtsw_repo)

get_ipython().run_cell_magic('time', '', '\nSTART_FROM = 13300+1 #index\nfor i, repo_name in enumerate(repos.iloc[START_FROM:]["repo"].values):\n    if ((START_FROM+i) % 100) == 0: \n        print(START_FROM+i,repo_name)\n        nx.write_gpickle(g, "github.BAK.gpickle")\n    try:\n        repo = client.get_repo(repo_name)\n        g.add_node(repo.name + \'(repo)\', \n                   type=\'repo\', \n                   lang=repo.language, \n                   owner=repo.owner.login,\n                   description=repo.description\n                  )\n        cache[\'repos\'][repo_name] = repo\n        \n        for contributor in repo.get_contributors():\n            g.add_node(contributor.login + \'(user)\', type=\'user\')\n            g.add_edge(contributor.login + \'(user)\', repo.name + \'(repo)\', type=\'contributors\')\n            cache[\'contributors\'][contributor.login] = contributor\n            for follower in contributor.get_followers():\n                if follower.login + \'(user)\' in g:\n                    g.add_edge(follower.login + \'(user)\', contributor.login + \'(user)\', \n                           type=\'follows\')\n                    cache[\'followers\'][follower.login] = follower\n        \n    except Exception, e: #ssl.SSLError\n        print >> sys.stderr, "Encountered an error fetching data for", \\\n                             repo_name, "Skipping."\n        print >> sys.stderr, e\n\n    #if ((START_FROM+i) % 100) == 0: print "Processed", START_FROM+i+1, " ", repo_name," nodes. Num nodes/edges in graph", \\\n    #      g.number_of_nodes(), "/", g.number_of_edges()\n    if client.rate_limiting < 400: print("Rate limit remaining", client.rate_limiting)\n    \nnx.write_gpickle(g, "github.BAK.gpickle")    \npickle.dump( cache, open( "cache.github.BAK.pickle", "wb" ) )')

len(cache["repos"])

pickle.dump( cache, open( "cache.github.pickle", "wb" ) )

nx.info(g)

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

