# special IPython command to prepare the notebook for matplotlib
get_ipython().magic('matplotlib inline')

import requests 
import StringIO
import zipfile
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting 

# If this module is not already installed, you may need to install it. 
# You can do this by typing 'pip install seaborn' in the command line
import seaborn as sns 

import sklearn
import sklearn.datasets
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.grid_search
import sklearn.neighbors
import sklearn.metrics

def getZIP(zipFileName):
    r = requests.get(zipFileName).content
    s = StringIO.StringIO(r)
    zf = zipfile.ZipFile(s, 'r') # Read in a list of zipped files
    return zf

url = 'http://seanlahman.com/files/database/lahman-csv_2014-02-14.zip'
zf = getZIP(url)
tablenames = zf.namelist()
print tablenames

teams = pd.read_csv(zf.open(tablenames[tablenames.index('Teams.csv')]))
players = pd.read_csv(zf.open(tablenames[tablenames.index('Batting.csv')]))
salaries = pd.read_csv(zf.open(tablenames[tablenames.index('Salaries.csv')]))
fielding = pd.read_csv(zf.open(tablenames[tablenames.index('Fielding.csv')]))
master = pd.read_csv(zf.open(tablenames[tablenames.index('Master.csv')]))

byPlayerID = salaries.groupby('playerID')['playerID','salary'].median()
medianSalaries = pd.merge(master[['playerID', 'nameFirst', 'nameLast']], byPlayerID,                   left_on='playerID', right_index = True, how="inner")
medianSalaries.head()

subTeams = teams[(teams['G'] == 162) & (teams['yearID'] > 1947)].copy()

subTeams["1B"] = subTeams.H - subTeams["2B"] - subTeams["3B"] - subTeams["HR"]
subTeams["PA"] = subTeams.BB + subTeams.AB

for col in ["1B","2B","3B","HR","BB"]:
    subTeams[col] = subTeams[col]/subTeams.PA
    
stats = subTeams[["teamID","yearID","W","1B","2B","3B","HR","BB"]].copy()
stats.head()

for col in ["1B","2B","3B","HR","BB"]:
    plt.scatter(stats.yearID, stats[col], c="g", alpha=0.5)
    plt.title(col)
    plt.xlabel('Year')
    plt.ylabel('Rate')
    plt.show()

stats.groupby('yearID')["1B","2B","3B","HR","BB"].mean().head()

def meanNormalizeRates(df):
        subRates = df[["1B","2B","3B","HR","BB"]]
        df[["1B","2B","3B","HR","BB"]] = subRates - subRates.mean(axis=0)
        return df

stats = stats.groupby('yearID').apply(meanNormalizeRates)

from sklearn import linear_model
clf = linear_model.LinearRegression()

stat_train = stats[stats.yearID < 2002]
stat_test = stats[stats.yearID >= 2002]

XX_train = stat_train[["1B","2B","3B","HR","BB"]].values
XX_test = stat_test[["1B","2B","3B","HR","BB"]].values

YY_train = stat_train.W.values
YY_test = stat_test.W.values
clf.fit(XX_train,YY_train)
clf.coef_

print("Mean squared error: %.2f"
      % np.mean((YY_test - clf.predict(XX_test)) ** 2))

subPlayers = players[(players.AB + players.BB > 500)  & (players.yearID > 1947)].copy()

subPlayers["1B"] = subPlayers.H - subPlayers["2B"] - subPlayers["3B"] - subPlayers["HR"]
subPlayers["PA"] = subPlayers.BB + subPlayers.AB

for col in ["1B","2B","3B","HR","BB"]:
    subPlayers[col] = subPlayers[col]/subPlayers.PA

# Create playerstats DataFrame
playerstats = subPlayers[["playerID","yearID","1B","2B","3B","HR","BB"]].copy()

playerstats = playerstats.groupby('yearID').apply(meanNormalizeRates)

playerstats.head()

def meanNormalizePlayerLS(df):
    df = df[['playerID', '1B','2B','3B','HR','BB']].mean()
    return df

def getyear(x):
    return int(x[0:4])

playerLS = playerstats.groupby('playerID').apply(meanNormalizePlayerLS).reset_index()

playerLS = master[["playerID","debut","finalGame"]].merge(playerLS, how='inner', on="playerID")
playerLS.head()

playerLS["debut"] = playerLS.debut.apply(getyear)
playerLS["finalGame"] = playerLS.finalGame.apply(getyear)
cols = list(playerLS.columns)
cols[1:3]=["minYear","maxYear"]
playerLS.columns = cols
playerLS.head()

avgRates = playerLS[["1B","2B","3B","HR","BB"]].values
playerLS["OPW"] = clf.predict(avgRates)
playerLS.head()

from collections import defaultdict

def find_pos(df):
    positions = df.POS
    d = defaultdict(int)
    for pos in positions:
        d[pos] += 1
    result = max(d.iteritems(), key=lambda x: x[1])
    return result[0]

positions_df = fielding.groupby("playerID").apply(find_pos)
positions_df = positions_df.reset_index()
positions_df = positions_df.rename(columns={0:"POS"})
playerLS_merged = positions_df.merge(playerLS, how='inner', on="playerID")
playerLS_merged = playerLS_merged.merge(medianSalaries, how='inner', on=['playerID'])
playerLS_merged.head()

active = playerLS_merged[(playerLS_merged["minYear"] <= 2002) &                          (playerLS_merged["maxYear"] >= 2003) &                          (playerLS_merged["maxYear"] - playerLS_merged["minYear"] >= 3)  ]
fig = plt.figure()
ax = fig.gca()
ax.scatter(active.salary/10**6, active.OPW, alpha=0.5, c='red')
ax.set_xscale('log')
ax.set_xlabel('Salary (in Millions) on log')
ax.set_ylabel('OPW')
ax.set_title('Relationship between Salary and Predicted Number of Wins')
plt.show()

def meanNormalizeOPW(df):
    tmp = df[['resid']] 
    df[['resid']]=tmp-tmp.mean(axis=0)
    return df

active['resid']=active['OPW']
active = active.groupby('POS').apply(meanNormalizeOPW)

Y = active.resid.values
X = np.log(active[["salary"]])

clf = linear_model.LinearRegression()
clf.fit(X,Y)

active['resid'] = Y - clf.predict(X)

active = active[active.resid >= 0]

def getMinSalary(s):
    return s["salary"].min()

minSalaryByPos = active.groupby('POS').apply(getMinSalary)
minSalaryByPos.sort(ascending=False)

posleft = list(minSalaryByPos.index)
print posleft

minSalaryByPos

moneyleft = 20*10**6

indexes=[]
    
for i in range(len(posleft)):
    
    # you need to have at least this much left to not go over in the next picks
    maxmoney = moneyleft - sum([minSalaryByPos[x] for x in posleft[1:] ])
    
    # consider only players in positions we have not selected
    index = [True if elem in posleft else False for elem in active.POS.values]
    left = active[index & (active.salary <= maxmoney)]
    
    # pick the one that stands out the most from what is expected given his salary
    j = left["resid"].argmax()
    indexes.append(j)
    
    # remove position we just filled from posleft
    posleft.remove(left.loc[j].POS)
    moneyleft = moneyleft - left.loc[j].salary
   
topPicks=active.loc[indexes,:]
topPicks=topPicks.sort(["OPW"],ascending=False)

moneyleft

topPicks

topPicks['salary'].sum()

round(topPicks['OPW'].mean())

def round1000(x):
    return np.round(x*1000)

topPicks[["1B","2B","3B", "HR","BB"]] = topPicks[["1B","2B","3B", "HR","BB"]].apply(round1000)
topPicks[["OPW"]] = np.round(topPicks[["OPW"]])
topPicks[["nameFirst","nameLast","POS","1B","2B","3B", "HR","BB","OPW","salary","minYear","maxYear"]]

