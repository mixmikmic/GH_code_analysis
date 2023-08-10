import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
import os
import datetime, time
import re

home_dir = "/Users/christopherallison/.virtualenvs/py_twi/results"
save_dir = "/Users/christopherallison/Documents/Coding/Gephi/twitter"

# Enter your search queries here
search_queries = "rstats python".split()

# Convert text to camel_case

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')

def convert(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()

# Remove empty mentions created through dataframe list comprehension
# i.e, ['twitter_user_A', None]

def clean_mentions(scr, men):
    temp = []
    for s, m in zip(scr, men):
        if m:
            for i, v in enumerate(m):
                temp.append([s, m[i]])
        else:
            pass
        
    return temp
        

# Remove edges for reply_to_tweet where no reply exists
# Need to add code to assign weight for duplicate edges - multiple mentions or replies to from same person
# Will return list on len 3 - node, target, weight

def clean_edges(edge_list):
    clean_edges = []

    for e in edge_list:
        if e[1] == None:
            pass
        else:
            clean_edges.append(e)
            
    return clean_edges

# Check for duplicates in edges and compute number of repeats as edge weight

def sort_and_weight(edges):
    temp = {}

    for i in edges:
        try:
            temp[str(i)]['weight'] += 1
        except KeyError:
            temp[str(i)] = {}
            temp[str(i)]['node'] = i[0]
            temp[str(i)]['target'] = i[1]
            temp[str(i)]['weight'] = 1

    return temp

def twitter_data_to_graph(search_query):
    
    for sq in search_query:
        
        fname = open(os.path.join(home_dir, 'search_{}.json'.format(sq)))
        
        # Load the JSON file
        df = pd.read_json(fname)
        
        # Swap rows & columns
        df2 = df.transpose()
        
        # Change index from Timestamp to integer
        df2.index = [t.value // 1 for t in df2.index]
        df2.head()
        
        # Pull hashtags from dict
        new_hash = []

        for h in df2.hashtags:
            n = []
            for i in h:
                n.append(i['text'].lower())
            new_hash.append(n)
            
        # Pull user names from dict
        new_mentions = []

        for m in df2.user_mentions:
            n = []
            for i in m:
                n.append(i['screen_name'])
            new_mentions.append(n)
            
        # Set up lists for node & edge creation
        texts = [text for text in df2.text]
        dates = [d for d in df2.date]
        replies = [r for r in df2.reply_to_tweet]
        user_replies = [u for u in df2.reply_to]
        mentions = [m for m in new_mentions]
        screen_names = [sn for sn in df2.user_screen_name]
        hashtags = [h for h in new_hash]
        retweets = [r for r in df2.retweet_count]
        id_str = [i for i in df2.id_str]
        quote_id_str = [q for q in df2.quoted_status_id_str]
        coords = [c for c in df2.coordinates]
        favorites = [f for f in df2.favorite_count]
        
        # Create one node for each tweet with embedded data - Graph N
        for i, n, d, h, r, t, q, c, f  in zip(id_str, screen_names, dates, hashtags, retweets, texts, quote_id_str, coords, favorites):
            N.add_node(i, attr_dict={
                    'name':n,
                    'date':d,
                    'hashtags':str(h),
                    'retweets':r,
                    'text':t, 
                    #'quoted_id':q,
                    #'coords':c,
                    #'favorites':f,
                })
        
        # Create edges list for each tweet & reply_to_tweet
        
        edges_r = [[n, r] for n,r in zip(id_str, replies)]
        edges_q = [[n, q] for n,q in zip(id_str, quote_id_str)]
        
        # Combine edges and prep - expanding multiple @-mentions FREX
        edges_all = edges_r + edges_q
        edges_all = clean_edges(edges_all)
        
        # Create edge weight as number of repeat replies & @-mentions
        
        edge_dict = sort_and_weight(edges_all)
        
        # Add edges for @mentions and replies to users
        for e in edge_dict:
            N.add_edge(edge_dict[e]['node'],
                       edge_dict[e]['target'],
                       weight=edge_dict[e]['weight'])
            
        # Insert Data analysis here
        print("Nodes: {}, Edges: {}".format(len(N.nodes()), len(N.edges())))
        
    # Write N graph in gexf for Gephi
    file_name = "{}_tweets_graph_{}.gexf".format(
        convert("_".join(search_queries)),
        datetime.datetime.now())
    
    nx.write_gexf(N, os.path.join(save_dir, file_name))
    

# Set up Graph
N = nx.DiGraph()

twitter_data_to_graph(search_queries)







































































