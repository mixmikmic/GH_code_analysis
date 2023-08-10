#Imports
get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import copy
import pfr
import time

# (we will use 2003-2014 for modeling, with 2002 providing additional information)
years = [2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014]
years= [str(i) for i in years]
data={}

for i in years:
    data[i] = pd.read_csv(i+'plays.csv')
    data[i]['year'] = int(i)

fulldf = pd.concat(data.values())
fulldf = fulldf.reset_index(drop=True)
fullplays = len(fulldf)

# create columns for if the team with the ball is the home team, as well as time elapsed in half
fulldf['isHome'] = fulldf['tm'] == fulldf['home']
fulldf['secsElapsedInHalf'] = np.remainder(fulldf['secsElapsedInGame'],1800)
fulldf['isLast3minHalf'] = fulldf['secsElapsedInHalf'] >= 1620

# create columns for half (1st half, 2nd half, overtime)
fulldf['is1stHalf'] = fulldf['quarter'] < 3
fulldf['is2ndHalf'] = (fulldf['quarter'] > 2) & (fulldf['quarter'] < 5)
fulldf['half'] = 2 - (fulldf['is1stHalf'])
for i,row in fulldf.iterrows():
    if row['quarter'] == 5:
        fulldf.loc[i,'half'] = 'OT'

# each team starts with 3 at each half (2 in overtime)
def TO(df):
    if df['half'] == 'OT':
        return 2
    else:
        return 3

fulldf['hm_TO_left'] = fulldf.apply(TO,axis=1)
fulldf['aw_TO_left'] = fulldf.apply(TO,axis=1)

# check each row to see if the row above was a timeout, and if so, then subtract 1 for each row below the timeout for the team timeouts remaining in the half (for the team that called the timeout)
for i in range(1,fullplays):
    if (fulldf.loc[i-1,'bsID'] == fulldf.loc[i,'bsID']) & (fulldf.loc[i-1,'half'] == fulldf.loc[i,'half']):
        if fulldf.loc[i-1,'isTimeout']:
            if fulldf.loc[i-1,'timeoutTeam'] == fulldf.loc[i-1,'home']:
                fulldf.loc[i,'hm_TO_left'] = fulldf.loc[i-1,'hm_TO_left'] - 1
                fulldf.loc[i,'aw_TO_left'] = fulldf.loc[i-1,'aw_TO_left']
            elif fulldf.loc[i-1,'timeoutTeam'] == fulldf.loc[i-1,'away']:
                fulldf.loc[i,'aw_TO_left'] = fulldf.loc[i-1,'aw_TO_left'] - 1
                fulldf.loc[i,'hm_TO_left'] = fulldf.loc[i-1,'hm_TO_left']
        else:
            fulldf.loc[i,'hm_TO_left'] = fulldf.loc[i-1,'hm_TO_left']
            fulldf.loc[i,'aw_TO_left'] = fulldf.loc[i-1,'aw_TO_left']

# sometimes a team gets an injury 4th timeout, so change any -1's to 0's
fulldf['hm_TO_left'] = fulldf['hm_TO_left'].apply(lambda h:h if h >= 0 else 0)
fulldf['aw_TO_left'] = fulldf['aw_TO_left'].apply(lambda a:a if a >= 0 else 0)

# only keeep run and pass plays from fulldf in the dataframe df
dfrunpass = fulldf[(fulldf['isPass'] == True) | (fulldf['isRun'] == True)]
df1 = dfrunpass.reset_index(drop=True)

# save all team names except NaN values
tms = df1.tm.unique()[:-1]

# delete rows that have NaN value in team (due to pro-reference site error)
df = df1[df1.tm.isin(tms)].reset_index(drop=True)

# get rid of 2-point conversions (no real need to predict, small sample) and plays where the team with the ball is unknown (rare)
df = df[~np.isnan(df.down)].reset_index(drop=True)

# Have to account for the fact that the website gives the score after the play, so subtract 6 for TD plays from the scoring team
def tmscore(df):
    if df['isHome']:
        if df['isTD'] == False:
            sc = df['pbp_score_hm']
        else:
            if df['exp_pts_after'] < 0:
                sc = df['pbp_score_hm']
            else:
                sc = df['pbp_score_hm'] - 6
    else:
        if df['isTD'] == False:
            sc = df['pbp_score_aw']
        else:
            if df['exp_pts_after'] < 0:
                sc = df['pbp_score_aw']
            else:
                sc = df['pbp_score_aw'] - 6
    return sc

# Have to account for the fact that the website gives the score after the play, so subtract 6 for TD plays from the scoring team
def oppscore(df):
    if df['isHome']:
        if df['isTD'] == False:
            sc = df['pbp_score_aw']
        else:
            if df['exp_pts_after'] < 0:
                sc = df['pbp_score_aw'] - 6
            else:
                sc = df['pbp_score_aw']
    else:
        if df['isTD'] == False:
            sc = df['pbp_score_hm']
        else:
            if df['exp_pts_after'] < 0:
                sc = df['pbp_score_hm'] - 6
            else:
                sc = df['pbp_score_hm']
    return sc
    
def tmwinprob(df):
    if df['isHome']:
        return df['home_wp']
    else:
        return 100.0 - df['home_wp']
    
def tmTOleft(df):
    if df['isHome']:
        return df['hm_TO_left']
    else:
        return df['aw_TO_left']
    
def oppTOleft(df):
    if df['isHome']:
        return df['aw_TO_left']
    else:
        return df['hm_TO_left']

# apply those functions to create the columns
df['tm_score'] = df.apply(tmscore,axis=1)
df['opp_score'] = df.apply(oppscore,axis=1)
df['tm_winprob'] = df.apply(tmwinprob,axis=1)
df['tm_TO_left'] = df.apply(tmTOleft,axis=1)
df['opp_TO_left'] = df.apply(oppTOleft,axis=1)

# create scoring marging column (team with ball - opponent) and additive scoring value
df['margin'] = df['tm_score'] - df['opp_score']
df['addscore'] = df['tm_score'] + df['opp_score']

# indicators if the team with the ball is in FG Range and/or RedZone
df['isFGRange'] = df['distToGoal'] <= 38
df['isRedZone'] = df['distToGoal'] <= 20

def dist1(y):
    if y < 4:
        return 'short'
    elif y < 8:
        return 'medium'
    elif y < 12:
        return 'long'
    else:
        return 'very long'

# create column for distance to go group, another column that concatenates year and team and another that concatenates year, team, down, and distance grouping
df['dist_to_1st'] = [dist1(yd) for yd in df.yds_to_go.values]
df['yeartm'] = [str(yr) + str(t) for yr,t in zip(df['year'],df['tm'])]
df['yeartmdowndist_to_1st'] = [str(yr) + str(t) + str(d) + str(d1) for yr,t,d,d1 in zip(df['year'],df['tm'],df['down'],df['dist_to_1st'])]

# create pass pct (generally) for each team and year
passpct = (df.groupby('yeartm')['isPass'].mean())

# create pass pct by down and distance grouping for each team and year
ddpasspct = (df.groupby('yeartmdowndist_to_1st')['isPass'].mean())

# create list with concatenated values for lookup of pass pct
lastyeartmdowndist_to_1st = [str(yr-1) + str(t) + str(d) + str(d1) for yr,t,d,d1 in zip(df['year'],df['tm'],df['down'],df['dist_to_1st'])]

# define function to get prior year pass pct
def getpass(p):
    # try to get the previous year's pass pct by down and dist grouping
    try:
        return ddpasspct[p]
    
    # if the grouping does not exist for the previous year for that team, then take that team's overall pass pct
    except KeyError:
        return passpct[p[:7]]

# create pass pct by down and distance grouping for each team lagged by one season (take last season's pct, no priors for 2002)
nanval = len(df[df.year==2002])
prevp = [np.nan for i in xrange(nanval)]
prevp.extend([getpass(passp) for passp in lastyeartmdowndist_to_1st[nanval:]])
df['tm_prev_yr_pass'] = prevp

dates = df.bsID.apply(lambda bID: pfr.boxscores.BoxScore(bID).date())
years, months, days = zip(*((d.year, d.month, d.day) for d in dates))
df['month'] = months
df['day'] = days
df = df.sort_values(['tm', 'year', 'month', 'day', 'secsElapsedInGame'],ascending=True)

# add gameNum column
tmgb = df.groupby(['year','tm'])
for tm, tmdf in tmgb:
    bsIDsInOrder = tmdf.bsID.unique()
    for i, bs in enumerate(bsIDsInOrder):
        df.loc[df.bsID == bs, 'gameNum'] = i+1

# use a cache dictionary for inSeasonPassPct to save time running the function
cache = {}

def inSeasonPassPct(row, df):
    # don't need 2002 for this
    if row['year'] == 2002:
        return np.nan
    
    tup = (row['year'], row['tm'], row['gameNum'])
    # if this value has already been computed, return it
    if tup in cache:
        return cache[tup]
    
    # otherwise, get previous year's pass pct if it is week one
    if row.gameNum == 1:
        val = passpct[str(row.year-1) + str(row.tm)]
    
    # otherwise, else get pass pct for the team in games before current game
    else:
        thisSeason = df[df.year == row['year']]
        prevGames = thisSeason[(thisSeason.tm == row.tm) & (thisSeason.gameNum < row.gameNum)]
        val = prevGames.isPass.mean()
    
    # store the value in the cache and return
    cache[tup] = val
    return val


def inGamePassPct(row, df):
    # don't need 2002 for this
    if row['year'] == 2002:
        return np.nan
    
    # use the team's in-season pass pct for the first play in a game
    firstTime = df.loc[(df.tm == row.tm) & (df.bsID == row.bsID), 'secsElapsedInGame'].iloc[0]
    if row.secsElapsedInGame == firstTime:
        return inSeasonPassPct(row, df)
    
    # get the team's in-game pass pct by calculating the mean of isPass for all plays prior to the current one
    else:
        thisGame = df[df.bsID == row.bsID]
        inGame = thisGame[(thisGame.tm == row.tm) & (thisGame.secsElapsedInGame < row.secsElapsedInGame)]
        return inGame.isPass.mean()

# create prior pass pct for each team in current season (taking into account all games before the current one), if it is week one, then use the prior pass pct for each team from the season before (the full season value)
df['tm_in_season_pass'] = df.apply(inSeasonPassPct, args=(df,), axis=1)

# create pass pct for each team within current game (taking into account all plays before the current one), but if it is the first play, then use the prior pass pct for that team from the game before (the full game value), unless it is week one, then use the pass pct for the team from the season before (the full season value)
df['tm_in_game_pass'] = df.apply(inGamePassPct, args=(df,), axis=1)

df["RESP"] = df["isPass"]*1.0
df["inFGRange"] = df["isFGRange"]*1.0
df["inRedZone"] = df["isRedZone"]*1.0
df["inLast3minHalf"] = df["isLast3minHalf"]*1.0
df["inDown1"] = (df["down"] == 1.)*1.0
df["inDown2"] = (df["down"] == 2.)*1.0
df["inDown3"] = (df["down"] == 3.)*1.0
df["inQuarter1"] = (df["quarter"] == 1.)*1.0
df["inQuarter2"] = (df["quarter"] == 2.)*1.0
df["inQuarter3"] = (df["quarter"] == 3.)*1.0
df["inQuarter4"] = (df["quarter"] == 4.)*1.0


# initialize the first values of prevWasPass column to 0 and StartofDrive to 1 (first play of first team's drive in data set, but sort first by year and game [bsID] and chronologically [secsElapsedInGame])
df = df.sort_values(['year','bsID','secsElapsedInGame'])
df.loc[0,'prevWasPass'] = 0.0
df.loc[0,'StartofDrive'] = 1.0

# iterate through the rows to fill in these columns
for i in range(1,len(df)):
    # if the current row is the same game and the same team has the ball as the play above
    if (df.loc[i-1,'bsID'] == df.loc[i,'bsID']) & (df.loc[i-1,'tm'] == df.loc[i,'tm']):
        df.loc[i,'prevWasPass'] = df.loc[i-1,'isPass']*1.0
        df.loc[i,'StartofDrive'] = 0.0
    # otherwise, it is the start of a new drive
    else:
        df.loc[i,'prevWasPass'] = 0.0
        df.loc[i,'StartofDrive'] = 1.0

df = df[(df.exp_pts_before != 0) | (df.exp_pts_after != 0)]

df[df['year']<2005].to_csv('0204plays.csv', index=False, encoding = 'utf-8')
df[(df['year']>2004) & (df['year']<2008)].to_csv('0507plays.csv', index=False, encoding = 'utf-8')
df[(df['year']>2007) & (df['year']<2011)].to_csv('0810plays.csv', index=False, encoding = 'utf-8')
df[df['year']>2010].to_csv('1114plays.csv', index=False, encoding = 'utf-8')

completedf1 = pd.concat([pd.read_csv('0204plays.csv'),pd.read_csv('0507plays.csv'),pd.read_csv('0810plays.csv'),pd.read_csv('1114plays.csv')])
completedf = completedf1[completedf1['year']>2002].reset_index(drop=True)

# define number of plays in df
plays = len(completedf)

