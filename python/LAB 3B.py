import numpy as np
import pandas as pd
import pip
pip.main(['install', 'requests'])
pip.main(["install","newspaper3k"])
import newspaper
from newspaper import Article
from  textblob import TextBlob

url = 'https://www.bloomberg.com/news/articles/2018-02-22/if-you-believe-quants-nothing-happened-in-markets-this-month'
article = newspaper.Article(url)
article.download()
article.parse()
article.title
article.nlp()
article.keywords
article.summary
blob2 = TextBlob(article.text)
article.summary

wordlist = pd.DataFrame()
ssList=[]
for t in blob2.sentences:
    ww = []
    for word, tag in t.tags:
        if tag in ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
            ww.append(word.lemmatize())
    ss = ' '.join(ww)
    ssList.append(ss.lower())

wordlist = wordlist.append(ssList, ignore_index=True)    

wordlist
len(blob2.sentences)
wordlist.to_csv('summary.csv')
wordlist

tweettext=df['text']
user= json_normalize(df['user'])
tweettext
name= user['screen_name']



