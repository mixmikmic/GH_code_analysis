get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly
import plotly.graph_objs as go
warnings.filterwarnings('ignore')
pylab.rcParams['figure.figsize'] = (12, 6)
plotly.offline.init_notebook_mode()
pd.set_option('display.float_format', lambda x: '%.4f' % x)

tweets = pd.read_csv('data/eem_hyg_tweets.csv', encoding='utf-8',
                   usecols=['index','username', 'retweets', 'favorites', 'polarity', 'ticker', 'p_pos', 'p_neg'])
usernames = pd.read_csv('data/twitter_usernames.csv', encoding='utf-8',
                   usecols=['screen_name', 'followers_count'])

tweets.head()

usernames.head()

data = tweets.merge(usernames, how='outer', left_on='username', right_on='screen_name')
data['date'] = pd.to_datetime(data['index'])
data.drop(['screen_name', 'username', 'index'], axis=1, inplace=True)
data.fillna(0, inplace=True)
data.set_index('date', inplace=True)
data.index = data.index.normalize()
data.head()

total = pd.DataFrame(data.groupby([pd.TimeGrouper('M', closed='right'), 'ticker'])['ticker'].count())
total.columns = ['monthly_tweet_count']
total_eem = total[total.index.get_level_values(1) == 'eem']
total_eem = total_eem.reset_index(level=1).resample('D', how='last', fill_method='bfill').set_index('ticker', append=True)
total_hyg = total[total.index.get_level_values(1) == 'hyg']
total_hyg = total_hyg.reset_index(level=1).resample('D', how='last', fill_method='bfill').set_index('ticker', append=True)
total = total_eem.append(total_hyg).sort_index()
total.head()

df = data.set_index('ticker', append=True).join(total)
df['tweet_count'] = 1
df['weight'] = df.tweet_count / df.monthly_tweet_count
df['p_pos_w'] = df.weight*df.p_pos
df['p_neg_w'] = df.weight*df.p_neg
df['polarity_w'] = df.weight*df.polarity
df = df.drop(['p_pos', 'p_neg', 'polarity'], axis=1)
df.loc['2015-01-01']

f = {'tweet_count':['sum'], 'retweets':['sum'], 'favorites':['sum'], 'polarity_w':['sum'],
     'p_pos_w':['sum'], 'p_neg_w':['sum'], 'followers_count':['sum'], 'monthly_tweet_count':['sum'],
     'weight':['sum']}
 
groups = df.reset_index(level=1).groupby([pd.TimeGrouper('M'),'ticker'])
df = groups.agg(f)
df.drop(['1970-01-31'], inplace=True)
df.drop(['weight', 'monthly_tweet_count'], axis=1, inplace=True)
df.columns = df.columns.get_level_values(0)
df.head()

eem_df = df.reset_index(level=1)[df.index.get_level_values(1) == 'eem']
hyg_df = df.reset_index(level=1)[df.index.get_level_values(1) == 'hyg']

eem_df.head()

hyg_df.head()

plt.title('EEM: Probability of Positive Sentiment Distribution')
plt.xlabel('Prob. of Positive')
plt.ylabel('Frequency')
eem_df['p_pos_w'].hist()

plt.title('HYG: Probability of Positive Sentiment Distribution')
plt.xlabel('Prob. of Positive')
plt.ylabel('Frequency')
hyg_df['p_pos_w'].hist()

eem_mean = np.full((1, len(eem_df.index)), eem_df['p_pos_w'].mean())
hyg_mean = np.full((1, len(hyg_df.index)), hyg_df['p_pos_w'].mean())

trace0 = go.Scatter(x=eem_df.index,
                    y=eem_df['p_pos_w'].values,
                    mode='markers',
                    name='EEM Sentiment',
                    marker=dict(size=eem_df['tweet_count'].values,
                                sizemode='area',
                                color='blue')
                   )

trace1 = go.Scatter(x=eem_df.index,
                    y=eem_mean[0],
                    name='EEM Mean Sentiment',
                    line=dict(color='black',
                                dash='dash')
                   )

trace2 = go.Scatter(x=hyg_df.index,
                    y=hyg_df['p_pos_w'].values,
                    mode='markers',
                    name='HYG Sentiment',
                    marker=dict(size=hyg_df['tweet_count'].values,
                                sizemode='area')
                   )

trace3 = go.Scatter(x=hyg_df.index,
                    y=hyg_mean[0],
                    name='HYG Mean Sentiment',
                    line=dict(color='grey',
                               dash='dash')
                   )

layout = go.Layout(title='Twitter Sentiment of EEM & HYG',
                   xaxis=dict(title='Time'),
                   yaxis=dict(title='Sentiment'))

# plot in notebook
plotly.offline.iplot({
    "data": [trace0, trace1, trace2, trace3],
    "layout": layout
}, show_link=False)

# html plot
# plotly.offline.plot({
#     "data": [trace0, trace1, trace2, trace3],
#     "layout": layout
# }, show_link=False)

