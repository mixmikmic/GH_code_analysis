import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
sns.set_style('darkgrid')

df = pd.read_csv("./datasets/fifa_goals.csv")

print df.shape
df.head()

df.info()

df.isnull().sum()
# there are no missing values to clean

def range_scale(val):
    new_val = val/df_score_per_game['totalscore'].max()
    if new_val < 0.1:
        new_val = 0.1
        return new_val
    else:
        return np.round(new_val, decimals=2)
    
df['totalscore'] = df['team1score'] + df['team2score']
df_score_per_game = df.groupby('game_id')['totalscore'].max().sort_values(ascending=False)
df_score_per_game = pd.DataFrame(df_score_per_game)
score_range = df_score_per_game['totalscore'].max() - df_score_per_game['totalscore'].min()
action = df_score_per_game['totalscore'].astype(float).map(range_scale) * 10
action = pd.DataFrame(action)
action.reset_index(inplace=True)
action.head()

#Create a mask column to check if there is a tie in score between Team 1 and Team 2
df['tie'] = df['team1score'] == df['team2score']

#Count the highest number of ties within each game
tie_true = df['tie'] == True
df_tie_true = df[tie_true].groupby('game_id')['tie'].agg(['count']).sort_values(['count'], ascending=False)
df_tie_true.head(10)

#Assign a value of 0 to games with no ties
tie_false = df['tie'] == False
df_tie_false = df[tie_false].drop_duplicates('game_id')
df_tie_false['count'] = 0

#Combine the df_tie_false with df_tie_true to get a new df of the number of ties per game
df_tie = pd.concat([df_tie_true['count'], df_tie_false['count']])
df_tie = pd.DataFrame(df_tie, columns=['count'])
df_tie['tiesscore'] = (df_tie['count']/3. * 9.) +1.
df_tie.reset_index(inplace=True)
df_tie

unique_player = df[['game_id','player']].groupby('game_id')['player'].unique()
count_ud = {}
for unique in unique_player:
    for u in unique:
        if u in count_ud.keys():
            count_ud[u] += 1
        else:
            count_ud[u] = 1
count_pdf = pd.DataFrame(count_ud.items())
count_pdf = count_pdf.sort_values(1, ascending=False)
count_pdf.head(10)

count_pdf = count_pdf.sort_values(1, ascending=False).reset_index(drop=True)
count_pdf['score']=0
count_pdf.rename(columns={0:'player', 1:'match_with_score'}, inplace=True)
for i in range(10):
    count_pdf.loc[i,'score'] = 10-i
df['player_score'] = df['player'].apply(lambda x: count_pdf.loc[count_pdf['player'] == x,'score'].iloc[0])
topten = df[['game_id','player_score']].groupby('game_id').sum().reset_index(drop=True).sort_values('player_score', ascending=False)

topten.reset_index(inplace=True)
topten['star_score'] = (topten['player_score']/30.*9.) + 1.

topten.head()

topten.tail()

excitement = pd.merge(action, df_tie, left_on='game_id', right_on='index')
excitement.drop(['index', 'count'], axis=1, inplace=True)
excitement = pd.merge(excitement, topten, left_on='game_id', right_on='index')
excitement.drop(['index', 'player_score'], axis=1, inplace=True)

excitement.head()

# calculate weighted excitement score
excitement['excitement'] = excitement['totalscore']*.4 + excitement['tiesscore']*.5 + excitement['star_score']*.1
excitement.sort_values('excitement', ascending=False, inplace=True)
excitement.head()

excitement.tail()

dfh= df.copy()

# There are + in minute col. at 90+, 45+ and 120+
# https://www.washingtonpost.com/news/fancy-stats/wp/2014/06/12/stoppage-time-in-soccer-how-it-works-and-what-it-means/?utm_term=.afacfa0de718
# the plus is when a goal happens during stoppage time -> arbitiarily assign a 0.5 to those so we can identify them later

dfh['minute'].replace('45+', '45.5', inplace=True)
dfh['minute'].replace('90+', '90.5', inplace=True)
dfh['minute'].replace('120+', '120.5', inplace=True)
dfh['minute']= dfh['minute'].astype('float')
dfh.dtypes

fig = plt.figure(figsize=(15,7))
ax = fig.gca()

sns.distplot(dfh['minute'], bins=24)
_ = plt.xlabel('minute')
_ = plt.ylabel('count of goals')
_ = plt.title('Number of goals scored at minute in game')

# define function to have pricing 
def pricing(excitement, slotnum=1):
    """given excitement score of game (1-10 scale, 10 most exciting), advert num (1=before, 2=halftime, 3=after)
    return advert price per minute"""
    assert 1 <= slotnum <= 3
    p = 2.5
    price = ((excitement**p)/(10.**p) * 7.5) + 0.5
    
    # adjust for slot
    if slotnum ==2:
        price = price *.9
    elif slotnum ==1:
        price = price * .8
    return price
    
pricing(9, 1)

# pricing mechanism
# Assume max possible price is $8M and min possible price is $0.5M
price = pd.DataFrame({'Score': range(1,11)})
price['Before Game'] = [pricing(x,1) for x in price['Score']]
price['During Game'] = [pricing(x,2) for x in price['Score']]
price['After Game'] = [pricing(x,3) for x in price['Score']]

fig = plt.figure(figsize=(10, 7))
ax = fig.gca()

price.plot(x='Score', ax=ax)
ax = plt.xlabel('Game Excitement Score')
ax = plt.ylabel('Advert Price per Minute ($M)')
ax = plt.title('Advert Pricing Structure')


