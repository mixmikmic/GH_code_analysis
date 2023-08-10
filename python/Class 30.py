get_ipython().magic('matplotlib inline')
import pandas as pd

gamelogs_df = pd.read_csv('GL2016.TXT')
gamelogs_df.head()

gamelogs_df = pd.read_csv('GL2016.TXT',header=None)
gamelogs_df.head()

gamelogs_df.columns

len(gamelogs_df.columns)

list(range(1,len(gamelogs_df.columns)+1))

gamelogs_df.columns = range(1,len(gamelogs_df.columns)+1)

gamelogs_df.head()

len(gamelogs_df)

gamelogs_df[3].head()

gamelogs_df['season_year'] = 2016

pd.to_datetime(gamelogs_df[1],format='%Y%m%d').head()

gamelogs_df['datetime'] = pd.to_datetime(gamelogs_df[1],format='%Y%m%d')

gamelogs_df.head()

from bs4 import BeautifulSoup, Comment
import requests

#headers = headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
#raw = requests.get('http://www.baseball-reference.com/leagues/MLB/2015-standings.shtml',headers=headers).text
raw = requests.get('http://www.baseball-reference.com/leagues/MLB/2016-standings.shtml').text
soup = BeautifulSoup(raw,'html.parser')

#table_html = soup.find_all('table',{'id':'expanded_standings_overall'})

# Apparently the data is hidden inside an HTML comment tag (?!)
pretend_raw = soup.find_all(text=lambda e: isinstance(e, Comment))[14]
pretend_soup = BeautifulSoup(pretend_raw,'lxml')
table_html = pretend_soup.find_all('table',{'id':'expanded_standings_overall'})[0]

# Once you have your table_html, pandas will read it in for you!
standings_df = pd.read_html(str(table_html),index_col=0)[0]
standings_df

example_dict = {'column_one':[1,2,3],
                'column_two':['a','b','c']
               }

example_df = pd.DataFrame(example_dict)
example_df

gamelogs_df[[1,4,7]].head()

gamelogs_df.ix[1]

gamelogs_df.ix[10:150]

gamelogs_df.ix[::5]

gamelogs_df.ix[[1,2,3,5,7,11,13]]

gamelogs_df.sort_values(10,ascending=False)[[1,4,7,10,11]]

gamelogs_df.sort_values(10,ascending=False)[[1,4,7,10,11]].head()

gamelogs_df.sort_values([10,11,1],ascending=False)[[1,4,7,10,11]]

len(gamelogs_df)

gamelogs_df[3].value_counts()

weekday_gametime_crosstab = pd.crosstab(gamelogs_df[3],gamelogs_df[13])
weekday_gametime_crosstab

weekdays_in_order = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
weekday_gametime_crosstab.ix[weekdays_in_order]

pd.crosstab(gamelogs_df[3],gamelogs_df[8])

weekday_gametime_league_crosstab = pd.crosstab(gamelogs_df[3],[gamelogs_df[8],gamelogs_df[13]])
weekday_gametime_league_crosstab.ix[weekdays_in_order]

minutes_count = gamelogs_df[19].value_counts().sort_index()
minutes_count.head()

ax = minutes_count.plot()
ax.set_xlabel('Minutes')
ax.set_ylabel('Count')

ax = gamelogs_df[19].plot(kind='hist',bins=50)
ax.set_xlabel('Minutes')
ax.set_ylabel('Count')

gamelogs_df[19] > 300

five_hour_games = gamelogs_df[gamelogs_df[19] > 300]
five_hour_games

standings_df.query('L > 90')

five_hour_games[[20,21]]

