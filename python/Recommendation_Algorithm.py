import pandas as pd
import json
import numpy as np
import spacy
import nltk
import re
import gensim

us_videos_df = pd.read_csv('./data/USvideos.csv')

us_videos_df.sort_values(by=['video_id', 'trending_date'], ascending=True, inplace=True)

us_videos_df.head()

us_grouped_videos = us_videos_df.groupby(['video_id']).last().reset_index()

us_grouped_videos.head()

with open('./data/US_category_id.json') as data_file:
    data = json.load(data_file)

categories = []
for item in data['items']:
    category = {}
    category['category_id'] = int(item['id'])
    category['title'] = item['snippet']['title']
    categories.append(category)

categories_df = pd.DataFrame(categories)
categories_df.head()

us_final_df = us_grouped_videos.merge(categories_df, on = ['category_id'])
us_final_df.rename(columns={'title_y': 'category', 'title_x': 'video_name'}, inplace=True)

us_final_df.tags[2]

def splitTags(tag_list):
    tag_list = tag_list.split('|')
    output = ''
    for tag in tag_list:
        output += tag
    return output

us_final_df['tags'] = us_final_df['tags'].apply(splitTags)

us_final_df.columns

#get rid of the punctuations and set all characters to lowercase
RE_PREPROCESS = r'\W+|\d+' #the regular expressions that matches all non-characters

#get rid of punctuation and make everything lowercase
#the code belows works by looping through the array of text
#for a given piece of text we invoke the `re.sub` command where we pass in the regular expression, a space ' ' to
#subsitute all the matching characters with
#we then invoke the `lower()` method on the output of the re.sub command
#to make all the remaining characters
#the cleaned document is then stored in a list
#once this list has been filed it is then stored in a numpy array

i = 0
def process_features(desc):
    try:
        return re.sub(RE_PREPROCESS, ' ', desc)
    except:
        return " "

us_final_df['video_features'] = us_final_df['tags'].astype(str) + us_final_df['video_name'].astype(str)                         + us_final_df['channel_title'].astype(str) + us_final_df['description'] + us_final_df['category']

us_final_df['video_features'] = us_final_df['video_features'].apply(process_features)

feature_corpus = us_final_df['video_features'].values

feature_corpus

feature_corpus = [comment.lower() for comment in feature_corpus]

parser = spacy.load('en')
processed_feature_corpus = [parser(feature) for feature in feature_corpus]

token_corpus = [nltk.word_tokenize(str(feature)) for feature in processed_feature_corpus]    

word2vec_model = gensim.models.Word2Vec(sentences=token_corpus, min_count=1, size = 32)

word2vec_model.most_similar(positive=['trump', 'president'])



