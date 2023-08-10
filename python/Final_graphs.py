import numpy as np
import pandas as pd
import re

#plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.grid_objs import Grid, Column #sliders
import plotly.graph_objs as go #horizontal bar charts
import plotly.plotly as py
from plotly import tools #side by side
init_notebook_mode(connected=True)

#LDA
import pyLDAvis
import pyLDAvis.gensim
from nltk.corpus import stopwords
stop = stopwords.words('english')
pyLDAvis.enable_notebook()
from gensim import corpora, models

from scipy import stats

import warnings
warnings.filterwarnings('ignore')

tweets = pd.read_csv('./finalprojdata/tweets.csv')
users = pd.read_csv('./finalprojdata/users.csv')
fulltweets = pd.read_csv('./finalprojdata/fulltweets2.csv')

#clean the date to a Year-Month format
users['Date'] = pd.to_datetime(users['created_at'])
users = users[pd.notnull(users['created_at'])]
users = users.drop_duplicates(subset=['id'])
users['Date'] = users['Date'].apply(lambda x: x.strftime('%Y-%m'))

usersname = pd.DataFrame(users.name.str.split(' ',1).tolist(),
                                   columns = ['first','last'])

usersnamesum = usersname.groupby('first',as_index=False).size().reset_index(name='counts')

usersnamesum = usersnamesum.sort_values('counts', ascending=False).head(20)

## side by side bar plot
#first names
firstnames = usersname.groupby('first',as_index=False).size().reset_index(name='counts')
firstnames = firstnames.sort_values('counts', ascending=False).head(20)
data1 = go.Bar(
            x=firstnames['counts'],
            y=firstnames['first'],
            orientation = 'h',
            name = 'First Name'
)

#lastnames
lastnames = usersname.groupby('last',as_index=False).size().reset_index(name='counts')
lastnames  = lastnames .sort_values('counts', ascending=False).head(20)
data2 = go.Bar(
            x=lastnames ['counts'],
            y=lastnames ['last'],
            orientation = 'h',
            name = 'Last Name'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('First Name','Last Name'))
fig.append_trace(data1, 1, 1)
fig.append_trace(data2, 1, 2)
fig['layout'].update(height=600, width=800, title='Totals of First and Last Names of Fake Accounts', titlefont=dict(size=25))
iplot(fig)

#get rid of the special characters
descrip = users.description.copy().astype(str)
descrip = descrip.str.replace('[^\w\s]','')
descrip = descrip.str.replace('[\\r|\\n|\\t|_]',' ')
descrip = descrip.str.strip()

fulltweets_descrip = fulltweets.copy()
fulltweets_descrip.descrip = descrip
fulltweets_descrip.descrip = fulltweets_descrip.descrip.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))

# compile sample documents into a list
doc_set = fulltweets_descrip.descrip.values.copy()
# loop through document list
texts = [text.split(' ') for text in doc_set]
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=30, id2word = dictionary)

data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
data

# group by Date, create a count and sort
userssum = users.groupby('Date',as_index=False).size().reset_index(name='counts')
userssum = userssum.sort_values('Date')

trace0 = Bar(
    name = "Accounts Created Over Time",
    x=userssum.Date,
    y=userssum.counts)
data = Data([trace0])

layout = Layout(
    title = "Accounts Made from 2009 to 2017",
    yaxis=dict(
        title='Number of Accounts Made',
        range=[0, 100],
        titlefont=dict(
            size=20,
        )
    ),
   
    xaxis = dict(
        title='Year',
        range = ['2009-01','2017-1'],
        titlefont=dict(
            size=20,
        )
    )
)
fig = Figure(data=data, layout=layout)
fig['layout'].update(titlefont=dict(size=25))
iplot(fig)

users.location.value_counts()

#create a line of best fit
slope, intercept, r_value, p_value, std_err = stats.linregress(users.followers_count,users.statuses_count)
line = slope*users.followers_count+intercept

trace = go.Scatter(
    x = users.followers_count,
    y = users.statuses_count,
    mode = 'markers',
    text = users.name,
    marker = dict(size = 10,
        color = 'blue',
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        ))
)

trace2 = go.Scatter(
                  x=users.followers_count,
                  y=line,
                  mode='lines',
                  marker=go.Marker(color='black'),
                  name='Fit'
                  )

data = [trace, trace2]

layout = Layout(
    title = "Russian Fake Account Followers and Tweet Count",
    xaxis=dict(
            title='Number of Followers',
            dtick=10000,
            range=[0, 100000],
            titlefont=dict(
            size=20,
        )
    ),
   
    yaxis = dict(
        title='Number of Tweets',
        range=[0, 70000],
        titlefont=dict(
            size=20,
        )
    ),
    
    annotations=[dict(y=20576,x=61609,xref='x',
            yref='y',text='Jenna Abrams'),
                dict(y=42438,x=60897,xref='x',
            yref='y',text='New York City Today'),
                dict(y=44882,x=27745,xref='x',
            yref='y',text='New Orleans Online'),
                dict(x=98412,y=30624,xref='x',
            yref='y',text='Максим Дементьев'),
                dict(y=1370,x=11518,xref='x',
            yref='y',text='Black News'),
                dict(y=57212,x=11370,xref='x',
            yref='y',text='РИА ФАН'),
                dict(y=28939,x=31729,xref='x',
            yref='y',text='Washington Online')]
)
fig = Figure(data=data, layout=layout)
fig['layout'].update(height=900, width=1200, titlefont=dict(size=20))
iplot(fig)

# heat map showing the days and hours of users
m = pd.pivot_table(fulltweets,values = 'user_key',index='created_strDayofweek',
                      columns='created_strMonth', aggfunc=len,fill_value=0, dropna=False)
z= m.as_matrix()

trace = Heatmap(z=z,
                x=[i for i in np.arange(0,24)],
                y=['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday'],
                colorscale = 'Jet'
               )
layout = Layout(
    title='Number of Tweets Per Day Per Month',
    xaxis=dict(
        nticks=24, 
        title = 'Month',
        titlefont=dict(
        size=20)),
    
    yaxis = dict(
    ),

)
data=[trace]
fig = Figure(data=data,layout=layout)
fig['layout'].update(titlefont=dict(size=25))
iplot(fig, filename='numberoftweets.html')

# LDA topics of actual tweets
dictionary = gensim.corpora.Dictionary.load('dictionary.dict')
corpus = gensim.corpora.MmCorpus('corpus.mm')
ldamodel = gensim.models.LdaModel.load('topic.model')

data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
data

subusers = fulltweets.groupby('user_key', as_index=False).mean().sort_values(by='followers_count', ascending=False).user_key[0:20].values

#create monthly aggregated data
subfulltweets2 = fulltweets[fulltweets.user_key.isin(subusers)][['user_key', 'followers_count', 'Subjectivity', 'Sentiment', 'statuses_count', 'created_str']]
subfulltweets2['Date'] = pd.to_datetime(subfulltweets2['created_str'])
subfulltweets2 = subfulltweets2[pd.notnull(subfulltweets2['created_str'])]
subfulltweets2['Date'] = subfulltweets2['Date'].apply(lambda x: x.strftime('%Y-%m')) #change to '%Y-%m-%d' for fine tuned

tweetcount = subfulltweets2.groupby('Date',as_index=False).size().reset_index(name='counts')
tweetcount = tweetcount.sort_values('Date', ascending=True)

trace0 = Scatter(
    name = "Tweets Made",
    x=tweetcount.Date,
    y=tweetcount.counts,
    line = dict(
        width = 4))
data = Data([trace0])

layout = Layout(
    title = "Number of Tweets by Top 20 Followed Fake Accounts",
    yaxis=dict(
        title='Number of Tweets',
        range=[0, 3500],
        titlefont=dict(
            size=20,
        )
    ),
   
    xaxis = dict(
        title='Year',
        range = ['2014-12','2017-5'],
        titlefont=dict(
            size=20,
        )
    )
)
fig = Figure(data=data, layout=layout)
fig['layout'].update(titlefont=dict(size=25))
iplot(fig)

#create daily aggregated data
subfulltweets3 = fulltweets[fulltweets.user_key.isin(subusers)][['text','user_key', 'followers_count', 'Subjectivity', 'Sentiment', 'statuses_count', 'created_str']]
subfulltweets3['Date'] = pd.to_datetime(subfulltweets3['created_str'])
subfulltweets3 = subfulltweets3[pd.notnull(subfulltweets3['created_str'])]
subfulltweets3['Date'] = subfulltweets3['Date'].apply(lambda x: x.strftime('%Y-%m-%d')) #change to '%Y-%m-%d' for fine tuned
tweetcount2 = subfulltweets3.groupby('Date',as_index=False).size().reset_index(name='counts')
tweetcount2 = tweetcount2.sort_values('Date', ascending=True)

trace0 = Scatter(
    name = "Tweets Made",
    x=tweetcount2.Date,
    y=tweetcount2.counts,
    line = dict(
        width = 2))
data = Data([trace0])

layout = Layout(
    title = "Daily Tweets by Top 20 Followed Fake Accounts",
    yaxis=dict(
        title='Number of Tweets',
        range=[0, 600],
        titlefont=dict(
            size=20,
        )
    ),
   
    xaxis = dict(
        title='Year',
        range = ['2014-12','2017-5'],
        titlefont=dict(
            size=20,
        )
    ),
    annotations=[dict(y=484,x='2016-10-17',xref='x',
            yref='y',text='#makemehateyouinonephrase'),
                dict(y=113,x='2016-08-04',xref='x',
            yref='y',text='#obamaswishlist'),
                dict(y=301,x='2016-12-07',xref='x',
            yref='y',text='#idrunforpresidentif'),
                dict(y=374,x='2016-10-05',xref='x',
            yref='y',text='#ruinadinnerinonephrase'),
                dict(y=330,x='2017-01-02',xref='x',
            yref='y',text='#survivialtips')]
)
fig = Figure(data=data, layout=layout)
fig['layout'].update(titlefont=dict(size=25))
iplot(fig)

#subfulltweets3[subfulltweets3['Date']=='2017-01-02'] #use to look at inidivdual dates

tweetsent = subfulltweets2.groupby('Date',as_index=False)['Sentiment','Subjectivity'].agg(np.mean)
tweetsent = tweetsent.sort_values('Date', ascending=True)

trace0 = Scatter(
    name = "Sentiment",
    x=tweetsent.Date,
    y=tweetsent.Sentiment,
    line = dict(
        width = 3))

trace1 = Scatter(
    name = "Subjectivity",
    x=tweetsent.Date,
    y=tweetsent.Subjectivity,
    line = dict(
        width = 3))

data = Data([trace0, trace1])

layout = Layout(
    title = "Subjectivity and Sentiment of Top 20 Fake Accounts",
    yaxis=dict(
        title='Value',
        range=[-1,1],
        titlefont=dict(
            size=20,
        )
    ),
   
    xaxis = dict(
        title='Year',
        range = ['2014-12','2017-5'],
        titlefont=dict(
            size=20,
        )
    )
)
fig = Figure(data=data, layout=layout)
fig['layout'].update(titlefont=dict(size=25))
iplot(fig)



