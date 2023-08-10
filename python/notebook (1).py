import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

data=pd.read_csv(r"../input/matches.csv")

#display the ipl data
data


data2008 = data[data['season'] == 2008]
data2009 = data[data['season'] == 2009]
data2010 = data[data['season'] == 2010]
data2011 = data[data['season'] == 2011]
data2012 = data[data['season'] == 2012]
data2013=data[data['season'] == 2013]
data2014=data[data['season'] == 2014]
data2015=data[data['season'] == 2015]
data2016=data[data['season'] == 2016]

#data.info()
#data2014.iloc[-4].winner
#data2014.describe()
#data2014.info()
#data2014.season.count() - used to count the number of matches in 2014 season.
data.info()

before_rows = data.shape
print(before_rows)

del data['umpire3']
data.dropna(inplace = True)

after_rows = data.shape
print(after_rows)
#decreased column number because of the umpire3 column.

data.isnull().any()
#no null values due to 'dropna' 

df = data2009[data2009['winner'] == 'Delhi Daredevils']
len(df)
win_csk = data2016[data2016['winner'] == 'Chennai Super Kings']
len(win_csk)
#len(df)
#data2012
## The value will be 0 as CSK didnt feature in CSK
df

data2010.iloc[0].winner

data2011.iloc[-1].winner

(data2008.iloc[-1].winner ,data2009.iloc[-1].winner ,data2010.iloc[-1].winner , data2011.iloc[-1].winner,data2012.iloc[-1].winner,data2013.iloc[-1].winner,data2014.iloc[-1].winner, data2015.iloc[-1].winner , data2016.iloc[-1].winner)
##the winner as a tuple



#(data.city.value_counts(normalize=True) *100).plot(kind='pie',title='Percentage Of Matches Played in Cities(All Seasons)',figsize=(12,8))
(data.winner.value_counts(normalize =True)*100).plot(kind = 'barh' , title='Percentage of matches won by Teams',figsize = (20,10))

#Percentage of toss decisions
(data.toss_decision.value_counts(normalize=True)*100).plot(kind='barh',title='Percentage of toss decisions(All Seasons)')


data2015.player_of_match.value_counts().head().plot(kind='barh',title="Top Players Become max times--\'Man of The Match'",grid=True) #No of the man of the match per player
#data2016.player_of_match.value_counts().head().plot(kind='barh',title="Top Players Become max times--\'Man of The Match'",grid=True) #No of the man of the match per player

#Number of Maximum Choosen Venue
venue = data.venue.value_counts()
data.venue.value_counts().plot(kind='bar',title='Fav Grounds' , figsize=(15,10) , grid =(15,10) ).legend(bbox_to_anchor=(1.2, 0.5))

#Winning Percent of teams at Chinnaswamy Stadium
chinna = data[data.venue=='M Chinnaswamy Stadium']['winner']
chinna.value_counts()
#chinna_win = data[data.venue=='M Chinnaswamy Stadium']['winner'].value_counts(normalize=True)*100
#chinna_win.plot(kind = 'line' , title='winning percent at Chinnaswamy' , figsize = (15,10) , grid=True).legend(bbox_to_anchor=(1.2, 0.5))

#df = data[data.venue=='M Chinnaswamy Stadium']['winner'].value_counts(normalize=True)*100
#df.plot(kind = 'barh' , title = 'winning percent of teams at Chinnaswamy' , figsize = (10,10) , grid = True)

#print (data.win_by_runs.mean() ) #Average win by runs
#print (data.win_by_wickets.mean()  ) #Average win by wicket
#data.describe()
#data.win_by_wickets.mean()
#data.win_by_runs.mean()
data.info()

#pd.crosstab(data.winner,data.season)
pd.crosstab( data2011.winner , data2011.toss_winner ).plot(kind = 'bar' , title = 'toss winner in each match in 2011 season' , figsize=(10,10))

pd.crosstab(data.winner,data.toss_decision).plot(kind='bar',title='Winning w.r.t toss decisions overall')

pd.crosstab(data2016.season,data2016.player_of_match).plot(kind='bar', title='Player of match in 2016 ').legend(bbox_to_anchor=(1.2, 0.5))

pd.crosstab(data2010.winner,data2010.toss_decision).plot(kind='bar',title='Winning w.r.t toss decisions in 2010')

pd.crosstab(data2011.winner,data2011.toss_decision).plot(kind='bar',title='Winning w.r.t toss decisions in 2011')

pd.crosstab(data2008.winner,data2008.city)

pd.crosstab(data2008.winner,data2008.city).plot(kind='bar',title='Winning w.r.t cities in 2008',figsize=(10,8))

fav_umpire=data.umpire1.value_counts().head(10)
#fav_umpire.plot(kind = 'barh')
fav_umpire

#From Seaborn
plt.subplots(figsize=(8,5))
sns.barplot(x=fav_umpire.values,y=fav_umpire.index,palette="Blues_d")

data.toss_winner.value_counts()

#Normal plot
data.toss_winner.value_counts().plot(kind='bar')

#By seaborn
plt.subplots(figsize=(8,5))
sns.barplot(x=data.toss_winner.value_counts().values,y=data.toss_winner.value_counts().index)



























































































































































