import praw

reddit = praw.Reddit(client_id='Pj5o8QpNXXJY9A',
                    client_secret='pQKMRBmhp0In48NoNvvktfRo2eA',
                    pasword = 'prawisgreat',
                    user_agent='Reddit Unlocked CS196 Project @ UIUC',
                    username='RedditUnlocked196')
news = reddit.subreddit('news')
for submission in news.top('year'):
    print(submission.url)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import plotly
plotly.tools.set_credentials_file(username='reddit_unlocked', api_key='gfnXKc7JvUKST4HRJyFX')
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *

top10news_df = pd.DataFrame({
    'title': (),
    'url' : (),
    'upvote_percentage': (),
    'year_posted': (),
    'month_posted': (),
    'day_posted': (),
    'is_self': (),
    'is_video': (),
    'media': (),
    'domain': (),
    'upvotes': (),
    'downvotes': (),
    'score': (),
    'views': ()
})
for submission in news.top('year', limit = 50):
    s = pd.Series([submission.title, submission.url, submission.upvote_ratio * 100,
                   datetime.utcfromtimestamp(submission.created_utc).year,
                  datetime.utcfromtimestamp(submission.created_utc).month,
                  datetime.utcfromtimestamp(submission.created_utc).day,
                  submission.is_self, submission.is_video, submission.media, submission.domain,
                   submission.score, submission.view_count,
                   int((submission.score * submission.upvote_ratio)/(2 * submission.upvote_ratio - 1))],
                  index=['title','url','upvote_percentage', 'year_posted', 'month_posted',
                         'day_posted', 'is_self', 'is_video', 'media', 'domain', 'score','views', 'upvotes'])
    top10news_df = top10news_df.append(s, ignore_index=True)
top10news_df['downvotes'] = top10news_df['upvotes'] - top10news_df['score']
top10news_df

from textblob import TextBlob, Word, Blobber
import newspaper
from newspaper import Article
import operator
import rake as rake
from datetime import datetime
rake_object = rake.Rake("SmartStoplist.txt", 1, 2, 1)

# This is the function in run_praw.py, so I'm using it as a reference for all my algs
def display_praw(name):
    reddit = praw.Reddit(client_id='Pj5o8QpNXXJY9A',
                         client_secret='pQKMRBmhp0In48NoNvvktfRo2eA',
                         password='prawisgreat',
                         user_agent='Reddit Unlocked CS196 Project @ UIUC',
                         username='RedditUnlocked196')

    subreddit = reddit.subreddit(name)

    threads_df = pd.DataFrame({
        'Title': (),
        'URL': (),
        'Upvote Ratio (%)': (),
        'Net Score': (),
        '# of Upvotes': (),
        '# of Downvotes': (),
        'Post Date': (),
        'Self Post?': (),
        'Video Post?': (),
        'Domain': ()
    })

    threads_df = threads_df[['Title', 'URL', 'Upvote Ratio (%)', 'Net Score', '# of Upvotes', '# of Downvotes',
                             'Post Date', 'Self Post?', 'Video Post?', 'Domain']]

    for thread in subreddit.top('year', limit=10): # TODO: change limit number when actually deploying program. 15 is the testing number.
        actualUps = int((thread.upvote_ratio * thread.score) / (thread.upvote_ratio * 2 - 1))
        actualDowns = actualUps - thread.score
        gather = pd.Series([thread.title, thread.url, thread.upvote_ratio * 100, thread.score,
                            actualUps, actualDowns, thread.created_utc,
                            thread.is_self, thread.is_video, thread.domain],
                           index=['Title', 'URL', 'Upvote Ratio (%)', 'Net Score', '# of Upvotes', '# of Downvotes',
                                  'Post Date', 'Self Post?', 'Video Post?', 'Domain'])
        threads_df = threads_df.append(gather, ignore_index=True)

    threads_dict = threads_df.to_dict(orient='records')

    for entry in threads_dict:
        if isinstance(str(entry['Post Date']), str):
            time = datetime.fromtimestamp(entry['Post Date'])
            formatTime = time.strftime('%b %d, %Y')
        else:
            formatTime = None

        entry['Post Date'] = formatTime

    return threads_dict


def get_keyword_dict():
    # Transforms dict returned by display_praw into DataFrame for working with
    top10news_df = pd.DataFrame.from_dict(display_praw('news'))

    words = {}

    ## NEWSPAPER STUFF HERE ##

    # Get keywords out of all articles
    for i in range(len(top10news_df)):
        #top10news_df.iloc[i]['url']
        myArticle = Article(top10news_df.iloc[i]['URL'])
        myArticle.download()
        myArticle.parse()
        myArticle.nlp()

        # Run sentiment analysis on each article, fetch subjectivity and polarity
        text = myArticle.text
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Get associated Reddit post info for each keyword, store in dictionary
        for keyword in myArticle.keywords:

            # Don't waste time with numeric keywords, skip them if they contain numbers
            if any(char.isdigit() for char in keyword):
                continue        

            if keyword not in words:
                words[keyword] = [keyword, 1, 
                                  top10news_df.iloc[i]['# of Upvotes'],
                                  top10news_df.iloc[i]["# of Downvotes"], 
                                  top10news_df.iloc[i]["Net Score"],
                                  subjectivity, polarity, 
                                  {(top10news_df.iloc[i]["Domain"]):1}]
            else:
                words[keyword][1] += 1
                words[keyword][2] += top10news_df.iloc[i]['# of Upvotes']
                words[currentWord][3] += int(top10news_df.iloc[i]['# of Downvotes'])
                words[currentWord][4] += int(top10news_df.iloc[i]['Net Score'])
                words[currentWord][5] = np.mean([subjectivity, words[currentWord][5]])
                words[currentWord][6] = np.mean([polarity, words[currentWord][6]])
                if top10news_df.iloc[i]["Domain"] in words[currentWord][7]:
                    words[currentWord][7][(top10news_df.iloc[i]["Domain"])] += 1
                else:
                    words[currentWord][7][top10news_df.iloc[i]["Domain"]] = 1

        ## RAKE STUFF HERE ##

        # Pull keywords from title strings
        for wordPair in rake_object.run(top10news_df.iloc[i]['Title']):
            currentWord = wordPair[0]

            # Don't waste time with numeric keywords, skip them if they contain numbers
            if any(char.isdigit() for char in currentWord):
                continue

            # Grab associated Reddit post data for each keyword, store in dictionary
            if currentWord not in words:
                words[currentWord] = [currentWord, 1, 
                                  top10news_df.iloc[i]['# of Upvotes'],
                                  top10news_df.iloc[i]["# of Downvotes"], 
                                  top10news_df.iloc[i]["Net Score"],
                                  subjectivity, polarity, 
                                  {(top10news_df.iloc[i]["Domain"]):1}]
            else:
                words[currentWord][1] += 1
                words[currentWord][2] += int(top10news_df.iloc[i]['# of Upvotes'])
                words[currentWord][3] += int(top10news_df.iloc[i]['# of Downvotes'])
                words[currentWord][4] += int(top10news_df.iloc[i]['Net Score'])
                if top10news_df.iloc[i]["Domain"] in words[currentWord][7]:
                    words[currentWord][7][(top10news_df.iloc[i]["Domain"])] += 1
                else:
                    words[currentWord][7][top10news_df.iloc[i]["Domain"]] = 1


    ### FOR GARY'S USE ###
    # Output dictionary is named 'words' #
    # Format is as such: #
    # key = keyword #
    # value = [Occurences, Upvotes, Downvotes, Score, Subjectivity, Polarity, Domain Dictionary] #
    
    return words
    
    

# For runtime comparison
start_time = datetime.now()

keywords_df = pd.DataFrame(get_keyword_dict(), index=['Keyword',
                                                      'Occurences',
                                                      'Upvotes', 
                                                      'Downvotes', 
                                                      "Score", 
                                                      "Subjectivity", 
                                                      "Polarity", 
                                                      "Domain"])

keywords_df = keywords_df.transpose().set_index('Keyword').sort_values('Occurences', ascending=False)
print(datetime.now() - start_time)
keywords_df

#code reference from https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html
pos = list(range(len(top10news_df['title'])))
width = .2

fig, ax = plt.subplots(figsize = (10,5))
plt.bar(pos, 
        #using df['upvotes'] data,
        top10news_df['upvotes'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label the first value in title
        label=top10news_df['title'][0]) 
plt.bar([p + width for p in pos],
       top10news_df['downvotes'],
        width,
        alpha = .5,
        color='#F78F1E',
        label=top10news_df['title'][1]
       )
ax.set_ylabel('votes')
ax.set_xlabel('post_score_rank')
ax.set_title('Reddit Post Upvotes and Downvotes')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(range(1,11))
plt.legend(['Upvotes', 'Downvotes'], loc='upper left')
plt.show()
py.iplot_mpl(fig)

#plotly interactive barchart testing
#code reference from https://plot.ly/python/ipython-notebook-tutorial/#plotting-interactive-maps
trace_upvotes = Bar(x=top10news_df.title,
                   y=top10news_df.upvotes,
                   name='Upvotes',
                   text = top10news_df.title,
                   textposition = 'auto',
                   marker=dict(color='#FFCDD2'))

trace_downvotes = Bar(x=top10news_df.title,
                     y=top10news_df.downvotes,
                     name='Downvotes',
                     marker=dict(color='#A2D5F2'))
data = [trace_upvotes, trace_downvotes]
layout = Layout(title="Reddit Post Upvotes and Downvotes",
               xaxis=dict(title='title'))
fig=Figure(data=data, layout=layout)
url = py.plot(fig, filename = 'barchart')
print(url)
#write function that takes in data frame and returns String of the url
#for the plotly interactive graph

trace1 = go.Scatter(
    x = top10news_df.month_posted,
    y = top10news_df.score,
    mode = 'markers+text',
    marker = dict(
        size = (top10news_df.upvotes + top10news_df.downvotes) / 200000 * 40,
        color = top10news_df.upvote_percentage,
        colorscale = 'Portland',
        showscale = True
    ),
    text = top10news_df.domain,
    textposition = 'bottom',
    textfont=dict(
        family='sans serif',
        size=18,
        color='#ff7f0e')
)
layout = go.Layout(
    title = 'Stats of top reddit/r/news posts',
    xaxis = dict(
        title = 'month_posted',
        ticks = 12,
    ),
    yaxis = dict(
        title = 'score',
        ticklen = 5,
    )
)
data = [trace1]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'reddit plot')

#Keyword incorporation
import newspaper
from newspaper import Article
import operator
import nltk
nltk.download('averaged_perceptron_tagger')

from textblob import TextBlob

######################
#                    #
#   keyword search   #
#                    #
######################

import newspaper
from newspaper import Article
import operator
import rake as rake
rake_object = rake.Rake("SmartStoplist.txt", 1, 2, 1)

words = {}

for i in range(len(top10news_df)):
    #top10news_df.iloc[i]['url']
    myArticle = Article(top10news_df.iloc[i]['url'])
    myArticle.download()
    myArticle.parse()
    myArticle.nlp()
    for keyword in myArticle.keywords:
        if keyword not in words:
            words[keyword] = [keyword, 1, top10news_df.iloc[i]['upvotes']]
        else:
            words[keyword][1] += 1
            words[keyword][2] += top10news_df.iloc[i]['upvotes']
    #RAKE STUFF HERE
    for wordPair in rake_object.run(top10news_df.iloc[i]['title']):
        currentWord = wordPair[0]
        #print(currentWord)
        if currentWord not in words:
            words[currentWord] = [currentWord, 1, top10news_df.iloc[i]['upvotes']]
        else:
            words[currentWord][1] += 1
            words[currentWord][2] += top10news_df.iloc[i]['upvotes']
keywords_df = pd.DataFrame(words, index=['Keyword','Occurences','Upvotes'])
keywords_df = keywords_df.transpose().set_index('Keyword')
keywords_df

words = []
for i in range(0, len(top10news_df['title'])):
    if (top10news_df['is_self'][i] == 0.0):
        a = Article(top10news_df['url'][i], language = 'en')
        a.download()
        a.parse()
        a.text
        a.nlp()
        for word in a.text:
            words.append(TextBlob(word))
print(words)
proper_df
for post in words:
    for word in post.tags:
        if word[1] == 'NNP':
            if keywords_df.contains(word[0]) and keywords_df['Occurences'][word[0]] > 1:
                proper_df.append(word[0])
proper_nouns





