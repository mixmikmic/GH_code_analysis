# first we want some Python tools to make our lives easier

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

with open("20170718 hashtag_damd uncleaned.csv") as fd:
    for row in fd.readlines()[:3]:
        print(row)

damd = pd.read_csv("20170718 hashtag_damd uncleaned.csv")

damd.columns

damd = pd.read_csv("20170718 hashtag_damd uncleaned.csv", index_col="tweet_id")
damd.head(3)

damd.hashtags.head()

def buildHashtagCooccurrenceGraph(tweets):
    g = nx.Graph(name="Hashtag co-occurrence bipartite")
    for tweet, hashtags in damd.hashtags.astype(str).map(lambda l: l.split(';')).items():
        g.add_node(tweet, Type="tweet_id")
        for hashtag in hashtags:
            g.add_edge(tweet, hashtag.lower())
    return g

g = buildHashtagCooccurrenceGraph(damd)

print(nx.info(g))

nx.write_gexf(g, "hashtag-cooccurrence-bipartite-with-python.gexf")

g_table2net = nx.read_gexf("hashtag-cooccurrence-bipartite-with-table2net.gexf")
print(nx.info(g_table2net))

# This algoritm is not guaranteed, but it is fast
nx.isomorphism.fast_could_be_isomorphic(g, g_table2net)

