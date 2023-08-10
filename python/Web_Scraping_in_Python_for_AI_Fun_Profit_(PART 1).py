from IPython.display import IFrame
iframe="https://tools.wmflabs.org/pageviews/?project=en.wikipedia.org&platform=all-access&agent=user&range=latest-20&pages=LeBron_James|Stephen_Curry#"
IFrame(iframe, width=900, height=900)

#Grab JSON results
import requests
lebron_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/LeBron_James/daily/2015070100/2017070500" 
res = requests.get(lebron_url)
lb_pageviews = res.json()

#clean up data and create Dataframe
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pageviews = []
timestamps = []    
names = []
wikipedia_handles = []

for record in lb_pageviews['items']:
            pageviews.append(record['views'])
            timestamps.append(record['timestamp'])
            names.append("LeBron")
            wikipedia_handles.append("LeBron_James")
data = {
        "names": names,
        "wikipedia_handles": wikipedia_handles,
        "pageviews": pageviews,
        "timestamps": timestamps 
    }

lb_df = pd.DataFrame(data)

# Make a simple plot
lb_df.plot(title="LeBron James Pageviews Wikipedia:  2015-217", figsize=(12, 4))

import pandas as pd
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
iris.head()

# Read it
nba_endorse_df = pd.read_csv('https://s3.amazonaws.com/aiwebscraping/socialpowernba/nba_2017_endorsement_full_stats.csv')
nba_endorse_df.describe()

# Transform it
nba_df_plot = nba_endorse_df[["PLAYER","ENDORSEMENT_MILLIONS", 
                              "WIKIPEDIA_PAGEVIEWS_10K", "WINS_RPM", 
                              "TWITTER_FAVORITE_COUNT_1K","SALARY_MILLIONS"]]

# Plot it with Plotly
import plotly.offline as py
from plotly.offline import init_notebook_mode
init_notebook_mode()
import plotly.graph_objs as go
trace1 = go.Bar(
    x=nba_df_plot['PLAYER'],
    y=nba_df_plot["ENDORSEMENT_MILLIONS"],
    name='Endorsements in Millions'
)
trace2 = go.Bar(
    x=nba_df_plot['PLAYER'],
    y=nba_df_plot["WIKIPEDIA_PAGEVIEWS_10K"],
    name='Wikipedia Pageviews'
)

trace3 = go.Bar(
    x=nba_df_plot['PLAYER'],
    y=nba_df_plot["WINS_RPM"],
    name='Wins Attributed to Player'
)

trace4 = go.Bar(
    x=nba_df_plot['PLAYER'],
    y=nba_df_plot["SALARY_MILLIONS"],
    name='Salary in Millions'
)

trace5 = go.Bar(
    x=nba_df_plot['PLAYER'],
    y=nba_df_plot["TWITTER_FAVORITE_COUNT_1K"],
    name='Twitter Favorite Count/1000'
)

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    barmode='group',
    title="2016-2017 NBA Season Endorsement and Social Power",
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')






