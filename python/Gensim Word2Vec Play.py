from gensim.models.word2vec import Word2Vec as w2
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn
from nltk import word_tokenize
from nltk.corpus import stopwords
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import numpy as np
import sklearn as sc
import os
import json

get_ipython().magic('matplotlib inline')

data_dir = '/Users/christopherallison/Documents/Coding/Data'
home_dir = '/Users/christopherallison/.virtualenvs/py_twi/streaming_results'

model = w2.load_word2vec_format(os.path.join(data_dir, 'GoogleNews-vectors-negative300.bin'), binary=True)

model.most_similar(positive=['king', 'man', 'wise'], negative=['old'], topn=5)

test_text = "invest people food".split()

model.most_similar(positive=test_text, negative=['money'], topn=5)

def load_tweets(target):
    # Output list of tweets
    tweets = []

    with open(os.path.join(home_dir, target)) as f:
        for data in f:

            result = json.loads(data)
            
            try:
                tweets.append(result['text'])
            except KeyError:
                continue
    
    return tweets



def process_tweets(tweets):
    # Output processed list of tweets
    # removed - if word not in stopwords.words('english') and word not in stopwords.words('french')

    texts = [[word for word in tweet.lower().split()] for tweet in tweets]
    
    # Remove words that only occur once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text] for text in texts]

    from pprint import pprint
    #pprint(texts)
    
    return texts

def analyze_sentiment(texts):
    # Output dictionary of tweets and sentiment scores for pleasure and activity
    
    tweet_sentiments = {}

    for i, text in enumerate(texts):
        pleasure = 0
        displeasure = 0
        active = 0
        passive = 0
        for word in text:
            try:
                pleasure += model.similarity('pleasure', word)
                displeasure += model.similarity('displeasure', word)
                active += model.similarity('active', word)
                passive += model.similarity('passive', word)
            except KeyError:
                continue
                
        tweet_sentiments[i] = {'text':" ".join(text), 'pleasure':pleasure - displeasure,
                           'active':active - passive}
        #print("Tweet {} Sentiment: pleasure {} active {}".format(i, pleasure - displeasure, active - passive))

    return tweet_sentiments

def prepare_coords(tweet_sentiments):
    # Output list of coordinates in tuple (x, y)
    coordinates = []

    for tweet in tweet_sentiments:
        x = tweet_sentiments[tweet]['pleasure'] - tweet_sentiments[tweet]['displeasure']
        y = tweet_sentiments[tweet]['active'] - tweet_sentiments[tweet]['passive']

        coordinates.append((x, y))
        
    return coordinates

def graph_tweets(tweet_sentiments, title):
    # Accepts dictionary of tweet sentiments & title, outputs graph and stats
    
    anger = 0
    excitement = 0
    content = 0
    sorrow = 0
    
    pleasure_sum = 0.0
    active_sum = 0.0
    
    plt.figure(figsize=(20,10))

    for tweet in tweet_sentiments:
        x = tweet_sentiments[tweet]['pleasure']
        y = tweet_sentiments[tweet]['active']
        
        pleasure_sum += x
        active_sum += y

        if x > 0 and y > 0:
            color = 'b'
            excitement += 1
        elif x < 0 and y > 0:
            color = 'r'
            anger += 1
        elif x > 0 and y < 0:
            color = 'g'
            content += 1
        else:
            color = 'y'
            sorrow += 1

        plt.plot(x, y, marker='o', markersize=8, color=color)
    
    plt.xlabel('pleasure')
    plt.ylabel('activity')
    plt.title('Sentiment Analysis of Tweets: {}'.format(title))
    
    plt.grid(True)
    blue_patch = mpatches.Patch(color='blue', label="Excitement")
    red_patch = mpatches.Patch(color='red', label='Anger')
    green_patch = mpatches.Patch(color='green', label='Content')
    yellow_patch = mpatches.Patch(color='yellow', label='Sorrow')
    
    plt.axvline(0)
    plt.axhline(0)

    first_legend = plt.legend(handles=[blue_patch], loc=1)
    second_legend = plt.legend(handles=[red_patch], loc=2)
    third_legend = plt.legend(handles=[green_patch], loc=4)
    fourth_legend = plt.legend(handles=[yellow_patch], loc=3)
    
    ax = plt.gca().add_artist(first_legend)
    ax = plt.gca().add_artist(second_legend)
    ax = plt.gca().add_artist(third_legend)
    
    print("Mood Breakdown for {}".format(title))
    print("Total Tweets: {}".format(len(tweet_sentiments)))
    print('Average Pleasure: {0:.2f}'.format(pleasure_sum / len(tweet_sentiments)))
    print('Average Active: {0:.2f}'.format(active_sum / len(tweet_sentiments)))
    print("*****")
    print('Excitement: {0:.2f}%'.format((excitement / len(tweet_sentiments))*100))
    print('Content: {0:.2f}%'.format((content / len(tweet_sentiments))*100))
    print('Anger: {0:.2f}%'.format((anger / len(tweet_sentiments))*100))
    print('Sorrow: {0:.2f}%\n'.format((sorrow / len(tweet_sentiments))*100))
    

# Plotly scatter
def plotly_graph_tweets(tweet_sentiments, title):
    
    x_coords = []
    y_coords = []
    texts = []
    color = []

    for tweet in tweet_sentiments:
        x = tweet_sentiments[tweet]['pleasure']
        y = tweet_sentiments[tweet]['active']
            
        x_coords.append(x)
        y_coords.append(y)
        texts.append(tweet_sentiments[tweet]['text'])
        
        if x > 0 and y > 0:
            color.append('blue')
        elif x < 0 and y > 0:
            color.append('red')
        elif x > 0 and y < 0:
            color.append('green')
        else:
            color.append('yellow')

    trace0 = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        name=title,
        text=texts,
        marker=dict(
            size=8,
            color=color,
        )
    )


    data = [trace0]
    layout = go.Layout(

        xaxis=dict(
            range=[-2.0, 2.0],
            autorange=False,
        ),

        yaxis=dict(
            range=[-2.0, 2.0],
            autorange=False,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='{} Plot'.format(title))

trudeau_tweets = load_tweets('trudeau_stream.json')
processed_trudeau_tweets = process_tweets(trudeau_tweets)
sentiment_trudeau = analyze_sentiment(processed_trudeau_tweets)

harper_tweets = load_tweets('harper_stream.json')
processed_harper_tweets = process_tweets(harper_tweets)
sentiment_harper = analyze_sentiment(processed_harper_tweets)

graph_tweets(sentiment_trudeau, 'Trudeau')



plotly_graph_tweets(sentiment_trudeau, 'Trudeau')

graph_tweets(sentiment_harper, "Harper")

mulcair_tweets = load_tweets('mulcair_stream.json')
processed_mulcair_tweets = process_tweets(mulcair_tweets)
sentiment_mulcair = analyze_sentiment(processed_mulcair_tweets)

graph_tweets(sentiment_mulcair, "mulcair")

plotly_graph_tweets(sentiment_mulcair, 'Mulcair')

cdnpoli_tweets = load_tweets('cdnpoli_stream.json')
processed_cdnpoli_tweets = process_tweets(cdnpoli_tweets)
sentiment_cdnpoli = analyze_sentiment(processed_cdnpoli_tweets)

graph_tweets(sentiment_cdnpoli, "Cdnpoli")

class MyTweets(object):
    def __init__(self, target):
        self.target = target
    
    def __iter__(self):
        for line in open(os.path.join(home_dir, self.target)):
            result = json.loads(line)
            text = result['text']
            yield dictionary.doc2bow(text.lower().split())









def getWordVecs(text):
    vecs = []
    for word in text:
        try:
            vecs.append(model[word].reshape(1,300))
        except KeyError:
            continue
    vecs = np.concatenate(vecs)
    return np.array(vecs, dtype='float')

tweet_text = []

for line in open(os.path.join(home_dir, 'harper_stream.json')):
    result = json.loads(line)
    text = result['text']
    tweet_text.append(text)

tweet_text



