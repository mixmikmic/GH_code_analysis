import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('combinedtweets.csv', encoding='utf-8')
df.head()

len(df)

from textblob import TextBlob

def sentiment_analysis(text):
    return TextBlob(text).sentiment

print df['text'][90]
sentiment_analysis(df['text'][90])

print df['text'][32]
sentiment_analysis(df['text'][32])

print df['text'][282]
sentiment_analysis(df['text'][282])

print df['text'][728]
sentiment_analysis(df['text'][728])

print df['text'][823984]
sentiment_analysis(df['text'][823984])

import sentlex
import sentlex.sentanalysis

def sentlex_analysis(text):
    SWN = sentlex.SWN3Lexicon()
    classifier = sentlex.sentanalysis.BasicDocSentiScore()
    classifier.classify_document(text, tagged=False, L=SWN, a=True, v=True, n=False, r=False, negation=False, verbose=False)
    return classifier.resultdata

sentlex_analysis(df['text'][0])

sentlex_analysis(df['text'][90])

sentlex_analysis(df['text'][32])

sentlex_analysis(df['text'][282])

sentlex_analysis(df['text'][728])

sentlex_analysis(df['text'][29385])

sentlex_analysis(df['text'][432673])

# split DataFrames into Republican and Democrat

df_repub = df[df.party == 'Republican']

df_repub_2016 = df_repub[(df_repub['created_at'] > '2015-12-31') & (df_repub['created_at'] < '2017-01-01')]

df_repub_2016.head(100)

len(df_repub)

len(df_repub_2016)

df_repub['text'].dtypes

df_repub['created_at'].dtypes

df_repub.iloc[2984]

df_dem = df[df.party == 'Democrat']

df_dem_2016 = df_dem[(df_dem['created_at'] > '2015-12-31') & (df_dem['created_at'] < '2017-01-01')]

df_dem.head(100)

len(df_dem)

len(df_dem_2016)

# interesting observation: Democrats and Republicans tweeted about the same amount in 2016!

df_dem.iloc[46657]

#transform text into array

tmp = [i.lower() for i in df_repub_2016['text']]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(tmp)

print type(X)

from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_topics=20, max_iter=5, random_state=1)
lda.fit(X)

tf_feature_names = vectorizer.get_feature_names()

print(type(tf_feature_names))
print(len(tf_feature_names))
print(tf_feature_names[:10])
print(tf_feature_names[-10:])

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {0}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

print_top_words(lda, tf_feature_names, n_top_words)



