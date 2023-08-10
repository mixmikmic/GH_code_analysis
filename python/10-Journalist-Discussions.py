import json
import sys
import os.path
import pandas as pd

get_ipython().magic('matplotlib inline')

# add penemue to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from utils import twiterate

tweetsfile = '../data/output/tweets.json'
usersfile = '../data/output/users.json'

u = json.load(open(usersfile))
users = {user['id_str']: user['description'] for user in u}

j_to_j = set()
j_to_o = set()
j_men_j = set()
j_men_o = set()

keywords = {"broadcaster", "journalists", "editor", "hack",
            "sub", "critic", "reporter", "journo", "commentator",
            "journalist", "columnist", "correspondent", "presenter",
            "producer", "features", "writing"}


def has_keyword(text):
    for word in keywords:
        if text != '' and text is not None:
            if word in text:
                return True
        
    return False

print('[Progress]')
print('[', end='')

for i, row in enumerate(open(tweetsfile)):
    tweet = json.loads(row)
    source = tweet['user']['id_str']
    source_description = tweet['user']['description']
    target = tweet['in_reply_to_user_id_str']
    
    if ('retweeted_status' not in tweet
    and has_keyword(source_description)):
        
        if (tweet['in_reply_to_status_id_str'] is not None
        and target is not None
        and target in users):
            
            target_description = users[target]
            if has_keyword(target_description):
                j_to_j.add((source, target))
            else:
                j_to_o.add((source, target))
            
        if len(tweet['entities']['user_mentions']) > 0:
            for mention in tweet['entities']['user_mentions']:
                if mention['id_str'] in users:
                    
                    target_description = users[mention['id_str']]
                    if has_keyword(target_description):
                        j_men_j.add((source, target))
                    else:
                        j_men_o.add((source, target))
                    
     
    if i % 1000 == 0:
        print('=', end='')

len_j_to_j = len(j_to_j)
len_j_to_o = len(j_to_o)
replies_total = len_j_to_j + len_j_to_o

len_j_men_j = len(j_men_j)
len_j_men_o = len(j_men_o)
mentions_total = len_j_men_j + len_j_men_o

print(']')

pd.DataFrame([[len_j_to_j, '%.2f' % ((len_j_to_j / replies_total) * 100)], 
              [len_j_to_o, '%.2f' % ((len_j_to_o / replies_total) * 100)],
              [len_j_men_j, '%.2f' % ((len_j_men_j / mentions_total) * 100)], 
              [len_j_men_o, '%.2f' % ((len_j_men_o / mentions_total) * 100)]],
             ['Journalist Replies to Journalist', 
              'Journalist Replies to Non Journalist',
              'Journalist Mentions Journalist',
              'Journalist Mentions Non Journalist'],
             ['Quantity', 'Percentage (%)'])



