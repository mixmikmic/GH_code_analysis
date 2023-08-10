get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.gridspec as gridspec
from numpy import random
import scipy.stats as scipy

from sklearn.metrics import accuracy_score

with sqlite3.connect('/home/sibi/Springboard/Data/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    matches = pd.read_sql_query("SELECT * from Match", con)
    leagues = pd.read_sql_query("SELECT * from League", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
    tempmatch = pd.read_sql_query("SELECT * from Match", con)

#Subsetting the five countries of interest
main_countries = ['England','France','Germany','Italy','Spain']
countries = countries[countries.name.isin(main_countries)]
leagues = countries.merge(leagues,on='id',suffixes=('', '_y'))
seasons = matches.season.unique()
leagues

def res(row):
    if row['home_team_goal'] == row['away_team_goal']:
        val = 0
    elif row['home_team_goal'] > row['away_team_goal']:
        val = 1
    else:
        val = -1
    return val

#Merge the leagues with corresponding matches
req_matches = matches[matches.league_id.isin(leagues['id'])]
req_matches = req_matches[['id','league_id','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal','season']]
req_matches["total_goals"] = req_matches['home_team_goal'] + req_matches['away_team_goal']
req_matches["result"] = req_matches.apply(res,axis = 1)
req_matches.dropna(inplace=True)
req_matches.head()

#Separating the leagues for plotting and further analysis
new_matches = pd.merge(req_matches,leagues,left_on='league_id', right_on='id')
new_matches = new_matches.drop(['id_x','id_y','country_id'],axis = 1)
english = new_matches[new_matches.name == "England"]
french = new_matches[new_matches.name == "France"]
italian = new_matches[new_matches.name == "Italy"]
spanish = new_matches[new_matches.name == "Spain"]
german = new_matches[new_matches.name == "Germany"]
# sum_goals = new_group_matches.home_team_goal.sum()
e = english.groupby('season')
f = french.groupby('season')
i = italian.groupby('season')
s = spanish.groupby('season')
g = german.groupby('season')
seasons

#Plotting total goals scored each season
fig = plt.figure(figsize=(10, 10))
plt.title("Total goals in each season")
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlabel("Season")
plt.ylabel("Total Goals Scored")
num_seasons = range(len(seasons))
plt.plot(num_seasons,e.total_goals.sum().values,label = "English Premier League", marker = 'o')
plt.plot(num_seasons,f.total_goals.sum().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,g.total_goals.sum().values,label = "German Bundesliga", marker = 'o')
plt.plot(num_seasons,i.total_goals.sum().values,label = "Italian Serie A", marker = 'o')
plt.plot(num_seasons,s.total_goals.sum().values,label = "Spanish La Liga", marker = 'o')
plt.legend()

#Plotting average goals scored each season
fig = plt.figure(figsize=(10, 10))
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlabel("Season")
plt.title("Average goals per game each season")
plt.ylabel("Average goals per game")
plt.plot(num_seasons,e.total_goals.mean().values,label = "English Premier League", marker = 'o')
plt.plot(num_seasons,f.total_goals.mean().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,g.total_goals.mean().values,label = "German Bundesliga", marker = 'o')
plt.plot(num_seasons,i.total_goals.mean().values,label = "Italian Serie A", marker = 'o')
plt.plot(num_seasons,s.total_goals.mean().values,label = "Spanish La Liga", marker = 'o')
#plt.xlim = (-20,20)
plt.legend(loc = 2)

#Plotting home/away scored each season
fig = plt.figure(figsize=(10, 10))
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.title('Home-away goal ratio each season')
plt.xlabel('Season')
plt.ylabel('Home-goals to Away-goals ratio')
plt.plot(num_seasons,e.home_team_goal.mean().values / e.away_team_goal.mean().values,label = "English Premier League", marker = 'o')
plt.plot(num_seasons,f.home_team_goal.mean().values / f.away_team_goal.mean().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,g.home_team_goal.mean().values / g.away_team_goal.mean().values,label = "German Bundesliga", marker = 'o')
plt.plot(num_seasons,i.home_team_goal.mean().values / i.away_team_goal.mean().values,label = "Italian Serie A", marker = 'o')
plt.plot(num_seasons,s.home_team_goal.mean().values / s.away_team_goal.mean().values,label = "Spanish La Liga", marker = 'o')
#plt.xlim = (-20,20),
plt.legend(loc = 1)

#Subsetting homewins vs homeloss from each of the leagues - ignoring draws.
e_hw = np.true_divide(english[english.result == 1].groupby('season').result.sum().values,english[english.result == -1].groupby('season').result.sum().values * -1)
f_hw = np.true_divide(french[french.result == 1].groupby('season').result.sum().values,french[french.result == -1].groupby('season').result.sum().values *-1)
g_hw = np.true_divide(german[german.result == 1].groupby('season').result.sum().values,german[german.result == -1].groupby('season').result.sum().values*-1)
i_hw = np.true_divide(italian[italian.result == 1].groupby('season').result.sum().values,italian[italian.result == -1].groupby('season').result.sum().values*-1)
s_hw = np.true_divide(spanish[spanish.result == 1].groupby('season').result.sum().values,spanish[spanish.result == -1].groupby('season').result.sum().values*-1)

#Plotting number of home wins vs home losses each season
fig = plt.figure(figsize=(10, 10))
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlim = (-20,20)
plt.ylim = (0,120)
plt.title("Number of home wins each vs home loss each season")
plt.xlabel("Season")
plt.ylabel("Home Wins vs loss")
plt.plot(num_seasons,e_hw,label = "English Premier League", marker = 'o')
plt.plot(num_seasons,f_hw,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,g_hw,label = "German Bundesliga", marker = 'o')
plt.plot(num_seasons,i_hw,label = "Italian Serie A", marker = 'o')
plt.plot(num_seasons,s_hw,label = "Spanish La Liga", marker = 'o')
plt.legend(loc = 1)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                '%d' % int(height),
                ha='center', va='bottom')

#New dataframe merging home_team names and matches.
matches_w_teams = pd.merge(new_matches,teams,left_on='home_team_api_id', right_on='team_api_id')
matches_w_teams = matches_w_teams.drop(['id','team_api_id','team_fifa_api_id'],axis = 1)
matches_w_teams = matches_w_teams.rename(columns={'team_long_name':'home_team_long_name','name_y':'league_name','name':'country_name'})
matches_w_teams.head(1)

#Color scheme for each country - from colourbrewer2.org
import matplotlib.patches as mpatches
colors = {'England':'#e41a1c', 'Spain':'#377eb8', 'Italy':'#4daf4a', 'France':'#984ea3', 'Germany':'#ff7f00'}
color = []

e = mpatches.Patch(color='#e41a1c', label='England')
s = mpatches.Patch(color='#377eb8', label='Spain')
it = mpatches.Patch(color='#4daf4a', label='Italy') #Facepalm note to self : never use i as it'll be used as an iterable for a for loop
f = mpatches.Patch(color='#984ea3', label='France')
g = mpatches.Patch(color='#ff7f00', label='Germany')

#Analysing teams in each league
top_n = 15
top_goal_scorers = matches_w_teams.groupby('home_team_long_name').total_goals.sum().sort_values(ascending = False)

for i in range(top_n):
    color.append([colors[t] for t in matches_w_teams[matches_w_teams.home_team_long_name == top_goal_scorers.head(top_n).index[i]].country_name.values][0])

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1,1,1)
rects = ax.bar(range(top_n), top_goal_scorers.head(top_n).values,align = "center",color = color)

ax.set_xticks(range(top_n))
ax.set_xticklabels(top_goal_scorers.head(top_n).index,rotation = "vertical")
ax.set_title("Top goal scorers at home")
ax.set_ylabel("Number of goals")
ax.set_xlabel("Team name")

ax.legend([it,g,e,s,f], colors.keys())
autolabel(rects)

#We get teams' home win vs home loss ratio
team_home_win = matches_w_teams[matches_w_teams.result == 1].groupby('home_team_long_name').result.sum().sort_index()
team_home_loss = matches_w_teams[matches_w_teams.result ==  -1].groupby('home_team_long_name').count().result.sort_index()
team_home_draw_loss = matches_w_teams[matches_w_teams.result !=  1].groupby('home_team_long_name').count().result.sort_index()
team_home_draw_win =  matches_w_teams[matches_w_teams.result !=  -1].groupby('home_team_long_name').count().result.sort_index()
np.setdiff1d(team_home_loss.index,team_home_win.index)

#We notice that a team "SpVgg Greuther Furth" never won a home game in the one season it played in Germany. Remove it.
team_home_loss = team_home_loss[team_home_loss.index.str.contains("SpV") == False]
team_home_draw_win = team_home_draw_win[team_home_draw_win.index.str.contains("SpV") == False]

team_home_wl_ratio = team_home_win /team_home_loss
team_home_wl_ratio = team_home_wl_ratio.sort_values(ascending = False)
#print team_home_wl_ratio.head()

team_home_wld_ratio = team_home_win / team_home_draw_loss
team_home_wld_ratio = team_home_wld_ratio.sort_values(ascending = False)
#team_home_wld_ratio.head()

team_home_wdl_ratio = team_home_draw_win / team_home_loss
team_home_wdl_ratio = team_home_wdl_ratio.sort_values(ascending = False)

#Plotting top_n ratios
fig = plt.figure(figsize = (20,20))
plt.style.use('ggplot')

colorwl = []
colorwld = []
colorwdl = []
for i in range(top_n):
    colorwl.append([colors[t] for t in matches_w_teams[matches_w_teams.home_team_long_name == team_home_wl_ratio.head(top_n).index[i]].country_name.values][0])
    colorwld.append([colors[t] for t in matches_w_teams[matches_w_teams.home_team_long_name == team_home_wld_ratio.head(top_n).index[i]].country_name.values][0])
    colorwdl.append([colors[t] for t in matches_w_teams[matches_w_teams.home_team_long_name == team_home_wdl_ratio.head(top_n).index[i]].country_name.values][0])

gs = gridspec.GridSpec(2, 4)
ax1 = fig.add_subplot(gs[0, :2],)
rects1 = ax1.bar(range(top_n), team_home_wl_ratio.head(top_n).values,align = "center", color = colorwl)
ax1.set_xticks(range(top_n))
ax1.set_xticklabels(team_home_wl_ratio.head(top_n).index,rotation = "vertical")
ax1.set_title("Team home to loss ratio (without draws)")
ax1.set_ylabel("Win to Loss ratio")
ax1.set_xlabel("Team names")
ax1.legend([it,g,e,s,f], ["Italy","Germany","England","Spain","France"])

ax2 = fig.add_subplot(gs[0, 2:])
rects2 = ax2.bar(range(top_n), team_home_wld_ratio.head(top_n).values,align = "center", color = colorwld)
ax2.set_xticks(range(top_n))
ax2.set_xticklabels(team_home_wld_ratio.head(top_n).index,rotation = "vertical")
ax2.set_title("Team home to loss ratio (with draws considered as a loss)")
ax2.set_ylabel("Win to (Loss or Draw) ratio")
ax2.set_xlabel("Team names")
ax2.legend([it,g,e,s,f], ["Italy","Germany","England","Spain","France"])

ax3 = fig.add_subplot(gs[1,1:3])
rects3 = ax3.bar(range(top_n), team_home_wdl_ratio.head(top_n).values,align = "center", color = colorwdl)
ax3.set_xticks(range(top_n))
ax3.set_xticklabels(team_home_wdl_ratio.head(top_n).index,rotation = "vertical")
ax3.set_title("Team home to loss ratio (with draws considered as a loss)")
ax3.set_ylabel("(Win or draw) to loss ratio")
ax3.set_xlabel("Team names")
ax3.legend([it,g,e,s,f], ["Italy","Germany","England","Spain","France"])
plt.tight_layout()

matches = matches[matches.league_id.isin(leagues.id)]
matches = matches[['id', 'country_id' ,'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','B365H', 'B365D' ,'B365A']]
matches.dropna(inplace=True)
# matches.head()

from scipy.stats import entropy


def match_entropy(row):
    odds = [row['B365H'],row['B365D'],row['B365A']]
    #change odds to probability
    probs = [1/o for o in odds]
    #normalize to sum to 1
    norm = sum(probs)
    probs = [p/norm for p in probs]
    return entropy(probs)

matches['entropy'] = matches.apply(match_entropy,axis=1)

mean_ent = matches.groupby(('season','league_id')).entropy.mean()
mean_ent = mean_ent.reset_index().pivot(index='season', columns='league_id', values='entropy')
mean_ent.columns = [leagues[leagues.id==x].name.values[0] for x in mean_ent.columns]
mean_ent.head(10)

ax = mean_ent.plot(figsize=(12,8),marker='o')
plt.title('Leagues Predictability', fontsize=16)
plt.xticks(rotation=50)

colors = [x.get_color() for x in ax.get_lines()]
colors_mapping = dict(zip(leagues.id,colors))

ax.set_xlabel('Mean Entropy')
plt.legend(loc='lower left')

ax.annotate('less predictable', xy=(7.3, 1.028), annotation_clip=False,fontsize=14,rotation='vertical')
ax.annotate('more predictable', xy=(7.3, 0.952), annotation_clip=False,fontsize=14,rotation='vertical')

from matplotlib.lines import Line2D


barcelona = teams[teams.team_long_name=='Barcelona'].team_api_id.values
offsets = [-0.16,-0.08,0,0.08,0.16]
offsets_mapping = dict(zip(colors_mapping.keys(),offsets))
y = []
x = []
c = []

i = -1
for season,season_df in matches.groupby('season'):
    i+=1
    for team,name in zip(teams.team_api_id,teams.team_long_name):
        team_df = season_df[(season_df.home_team_api_id==team)|(season_df.away_team_api_id==team)]
        team_entropy = team_df.entropy.mean()
        if team_entropy>0:
            league_id = team_df.league_id.values[0]
            x.append(i+offsets_mapping[league_id])
            y.append(team_entropy)
            c.append(colors_mapping[league_id])


plt.figure(figsize=(14,8))

plt.scatter(x,y,color=c,s=[40]*len(x))
plt.title('Teams Predictability', fontsize=16)

ax = plt.gca()
plt.xlim = (-0.5,7.5)
plt.xticks(np.arange(0,8,1),rotation=50)
ax.set_ylabel("Entropy")

ax.set_xticklabels(mean_ent.index)
for i in range(7):
    ax.axvline(x=0.5+i,ls='--',c='w')
ax.yaxis.grid(False)
ax.xaxis.grid(False)

circles = []
labels = []
for league_id,name in zip(leagues.id,leagues.name):
    labels.append(name)
    circles.append(Line2D([0], [0], linestyle="none", marker="o", markersize=6, markerfacecolor=colors_mapping[league_id]))
plt.legend(circles, labels, numpoints=3, loc=(0.005,0.02))


ax.annotate('', xytext=(7.65, 0.93),xy=(7.65, 1.1), arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1))
ax.annotate('', xytext=(7.65, 0.77),xy=(7.65, 0.6), arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1))

ax.annotate('less predictable', xy=(7.75, 1.05), annotation_clip=False,fontsize=14,rotation='vertical')
ax.annotate('more predictable', xy=(7.75, 0.73), annotation_clip=False,fontsize=14,rotation='vertical')

#add labels
ax.annotate('Barcelona', xy=(6.55, 0.634),fontsize=9)
ax.annotate('B. Munich', xy=(6.5, 0.655),fontsize=9)
ax.annotate('Real Madrid', xy=(6.51, 0.731),fontsize=9)
ax.annotate('PSG', xy=(6.93, 0.78),fontsize=9)

df = pd.read_csv("../Data/England/E0_13.csv")
df_14 = pd.read_csv("../Data/England/E0_14.csv")

res_13 = df.ix[:,:23]
res_13 = res_13.drop(['Div','Date','Referee'],axis=1)
res_14 = df_14.ix[:,:23]
res_14 = res_14.drop(['Div','Date','Referee'],axis=1)
bet_13 = df.ix[:,23:]

#Team, Home Goals Score, Away Goals Score, Attack Strength, Home Goals Conceded, Away Goals Conceded, Defensive Strength
table_13 = pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS'))

avg_home_scored_13 = res_13.FTHG.sum() / 380.0
avg_away_scored_13 = res_13.FTAG.sum() / 380.0
avg_home_conceded_13 = avg_away_scored_13
avg_away_conceded_13 = avg_home_scored_13
print "Average number of goals at home",avg_home_scored_13
print "Average number of goals away", avg_away_scored_13
print "Average number of goals conceded at home",avg_away_conceded_13
print "Average number of goals conceded away",avg_home_conceded_13

res_home = res_13.groupby('HomeTeam')
res_away = res_13.groupby('AwayTeam')

table_13.Team = res_home.HomeTeam.all().values
table_13.HGS = res_home.FTHG.sum().values
table_13.HGC = res_home.FTAG.sum().values
table_13.AGS = res_away.FTAG.sum().values
table_13.AGC = res_away.FTHG.sum().values
table_13.HAS = (table_13.HGS / 19.0) / avg_home_scored_13
table_13.AAS = (table_13.AGS / 19.0) / avg_away_scored_13
table_13.HDS = (table_13.HGC / 19.0) / avg_home_conceded_13
table_13.ADS = (table_13.AGC / 19.0) / avg_away_conceded_13

#Team, Home Goals Score, Away Goals Score, Attack Strength, Home Goals Conceded, Away Goals Conceded, Defensive Strength
table_13.head()

#Expected number of goals based on the average poisson probability
def exp_goals(mean):
    max_pmf = 0;
    for i in xrange(7):
        pmf = scipy.distributions.poisson.pmf(i,mean) * 100 
        if pmf > max_pmf:
            max_pmf = pmf
            goals = i
    return goals

test_13 = res_13.ix[:,0:5]
test_13.head()
test_14 = res_14.ix[:,0:5]
test_14.head()

table_13[table_13['Team'] == 'Arsenal']
test_14['ER'] = ''
test_14 = test_14.drop(test_14.index[380],axis=0)

results = []
for index, row in test_13.iterrows():

    home_team = table_13[table_13['Team'] == row['HomeTeam']]
    away_team = table_13[table_13['Team'] == row['AwayTeam']]
    #print "Home : ", home_team.HAS.values, "Away: ", away_team.AAS.
    if row.HomeTeam not in ['Leicester', 'QPR', 'Burnley'] and row.AwayTeam not in ['Leicester', 'QPR', 'Burnley']:
        EH = home_team.HAS.values * away_team.ADS.values * avg_home_scored_13
        EA = home_team.HDS.values * away_team.AAS.values * avg_home_conceded_13
        #print row.HomeTeam, row.AwayTeam
        if EH[0] > EA[0]:
            results.append('H')
        else:
            results.append('A')
    else:
        results.append('D')
test_13['ER'] = results
accuracy_score(test_13['ER'],test_13['FTR'])

team_1 = 'Man United'
team_2 = 'Cardiff'

home_team = table_13[table_13['Team'] == team_1]
away_team = table_13[table_13['Team'] == team_2]
EH = home_team.HAS.values * away_team.ADS.values * avg_home_scored_13
EA = home_team.HDS.values * away_team.AAS.values * avg_home_conceded_13

def exp_goals_prob(mean):
    max_pmf = 0;
    prob = []
    for i in xrange(0,6):
        pmf = scipy.distributions.poisson.pmf(i,mean) * 100 
        prob.append(pmf[0])
    return prob

prob_goals = pd.DataFrame(columns=['Team','0','1','2','3','4','5'])
home_team_prob = exp_goals_prob(EH)
away_team_prob = exp_goals_prob(EA)

prob_goals.loc[0,1:] = home_team_prob
prob_goals.loc[1,1:] = away_team_prob
prob_goals.iloc[0,0] = team_1
prob_goals.iloc[1,0] = team_2
prob_goals

feature_table = df.ix[:,:23]
feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST','HC','AC']]
f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for index,row in feature_table.iterrows():
    f_HAS.append(table_13[table_13['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table_13[table_13['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table_13[table_13['Team'] == row['HomeTeam']]['AAS'].values[0])
    f_ADS.append(table_13[table_13['Team'] == row['HomeTeam']]['ADS'].values[0])
    
feature_table['HAS'] = f_HAS
feature_table['HDS'] = f_HDS
feature_table['AAS'] = f_AAS
feature_table['ADS'] = f_ADS

def transformResult(row):
    if(row.FTR == 'H'):
        return 1
    elif(row.FTR == 'A'):
        return -1
    else:
        return 0

feature_table["Result"] = feature_table.apply(lambda row: transformResult(row),axis=1)

X_train = feature_table[['HAS','HDS','AAS','ADS']]
X_train_2 = feature_table[['HAS','HDS','AAS','ADS','HST','AST','HC','AC']]
y_train = feature_table['Result']

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

clf = [DecisionTreeClassifier(),RandomForestClassifier(), KNeighborsClassifier(n_neighbors=17), MultinomialNB(),XGBClassifier()]
labels = ['D-Tree','RF','KNN', 'Naive Bayes','XGBoost']
mean_scores = []
mean_scores_2 = []
cms = []
for i in xrange(0,5):
    clf[i].fit(X_train,y_train)
    clf[i].fit(X_train_2,y_train)

    scores = cross_val_score(clf[i], X_train, y_train, cv=10)
    scores_2 = cross_val_score(clf[i], X_train_2, y_train, cv=10)
    print labels[i]," : ", scores.mean(), " : ", scores_2.mean()
    
    mean_scores.append(scores.mean())  
    mean_scores_2.append(scores_2.mean())


fig = plt.figure(figsize = (12,10))
ax = fig.add_subplot(2,2,1)
ax.bar(xrange(0,5),mean_scores,align='center');

ax.set_xticks(range(5));

ax2 = fig.add_subplot(2,2,2);
ax2.bar(xrange(0,5),mean_scores_2,align='center');

ax.set_xticks(range(5));
ax.set_ylim(0,1);
ax.set_xticklabels(labels);

ax2.set_xticks(range(5))
ax2.set_ylim(0,1);
ax2.set_xticklabels(labels);

#Plotting top_n ratios
fig = plt.figure(figsize = (10,8))
plt.style.use('ggplot')
gs = gridspec.GridSpec(2, 4)

#RandomForest
ax1 = fig.add_subplot(gs[0, :2],)
y_pred = clf[1].predict(X_train_2)
confusion_matrix(y_train, y_pred)
conf = pd.DataFrame(confusion_matrix(y_train, y_pred),columns=("Home Win","Draw","Away Win"),index=("Home Win", "Draw","Away Win"))
sns.heatmap(conf,annot=True,fmt='d')
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Random Forest",fontsize=20)

ax2 = fig.add_subplot(gs[0, 2:])
y_pred = clf[2].predict(X_train_2)
confusion_matrix(y_train, y_pred)
conf = pd.DataFrame(confusion_matrix(y_train, y_pred),columns=("Home Win","Draw","Away Win"),index=("Home Win", "Draw","Away Win"))
sns.heatmap(conf,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN",fontsize=20)

ax3 = fig.add_subplot(gs[1,1:3])
y_pred = clf[3].predict(X_train_2)
confusion_matrix(y_train, y_pred)
conf = pd.DataFrame(confusion_matrix(y_train, y_pred),columns=("Home Win","Draw","Away Win"),index=("Home Win", "Draw","Away Win"))
sns.heatmap(conf,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes",fontsize=20)

plt.tight_layout()



