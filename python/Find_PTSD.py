import pandas as pd
import json

import numpy as np
import matplotlib.pyplot as plt

import pylab
import nltk
import operator 
from collections import Counter
import regex as re

json_tweet_btp='C:/Users/Robin/Dropbox/Travaux_&_Rapports_Stages/UbicomLab-GATech/MeToo/code/Data_tweets/clean_tweet.json'
json_btp_com='C:/Users/Robin/Dropbox/Travaux_&_Rapports_Stages/UbicomLab-GATech/MeToo/code/balancetonporc_com/strories_balancetonporc_com.json'
json_metoo='C:/Users/Robin/Dropbox/Travaux_&_Rapports_Stages/UbicomLab-GATech/MeToo/code/Data-metoo/clean_tweet_metoo.json'

df_story = pd.read_json(json_btp_com, orient="columns")
df_tweet = pd.read_json(json_tweet_btp, orient="columns")
df_metoo= pd.read_json(json_metoo, orient="columns")


df_tweet

# create text file with one tweet
import os
newpath = r'C:\Users\Robin\TweetsBTP' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(len(df_tweet['text'])):
    filename='TweetsBTP/tweet'+str(i)+"_"+str(df_tweet['id'].iloc[i])+'.txt'
    file = open(filename,'w',encoding='utf-8')
    #for j in range(len(doc_stories_clean[i])):
    text=df_tweet['text'].iloc[i]
    file.write(text)
    file.write('\n')
    file.close()

### win some time and delete stop word
import stop_words
sw = stop_words.get_stop_words('fr')+["""’"""]+['…']+['a']+['les']+['ça']+['cest']+['jai']

### key word for PTSD, depression
PTSD= ["ptsd","diagnos","tspt","syndrome","syndrom","sindrom","traumatique","sspt","anxiete","trouble","stress",
       "stres", ]
depression=["depression","depresion","deprime","deprimee","depressif"]
PTSD_en=['PTSD','ptsd','stress','anxiety','depression','syndrom','trauma','post-trauma']
HS=PTSD+depression
### function to find the relevant tweet and the story
def find_word(df,word):
    L=[]
    count_search = Counter()
    for i in range(len(df['text'])):
        text=nltk.word_tokenize(str.lower(df['text'].iloc[i]))
        terms_only = [term for term in text if term not in sw]
        for j in range(len(word)):
            if word[j] in terms_only:
                #count_search.update(terms_only)
                L.append(i)
    L=set(L)
    #for k in range(len(L)):
        #print(df['text'].iloc[L[k]],"/n")
    print("There are ",len(L)," users concerned: ",L, "soit",len(L)/len(df['text'])*100,"%")
    return(L)
        
    
    
    

L=find_word(df_metoo,PTSD_en)
print ("There are ",len(L)," users concerned: ",L, "soit",len(L)/len(df_metoo['text'])*100,"%")
#for i in find_word(df_metoo,PTSD_en):
#  #
#

