import numpy as np
import pandas as pd
import math

df_1 = pd.read_csv('NCAA_FCS_Results_2013_.csv')
df_2 = pd.read_csv('NCAA_FCS_Results_2014_.csv')
df_3 = pd.read_csv('NCAA_FCS_Results_2015_.csv')
df_4 = pd.read_csv('NCAA_FCS_Results_2016_.csv')

df = pd.concat([df_1,df_2,df_3,df_4],ignore_index=True)

df.head()

# fill missing data with 0
df = df.fillna(0)

df.head()

# update the weights based on a factorial scheme
df['weights'] = (df['year']-2012)
df['weights'] = df['weights'].apply(lambda x: math.factorial(x))

df['scorediff'] = (df['teamscore']-df['oppscore'])

df.head()

df['location'] = df['location'].replace('V',-1)
df['location'] = df['location'].replace('N',0)
df['location'] = df['location'].replace('H',1)

df.head()

# create dummy variables, need to do this in python b/c does not handle automatically like R
team_dummies = pd.get_dummies(df.team, prefix='team')
opponent_dummies = pd.get_dummies(df.opponent, prefix='opponent')

df = pd.concat([df, team_dummies, opponent_dummies], axis=1)

df.head()

# make the training data
X = df.drop(['year','month','day','team','opponent','teamscore','oppscore','D1','OT','weights','scorediff'], axis=1)
y = df['scorediff']
weights = df['weights']

X.head()

y.head()

weights.head()

from sklearn.linear_model import Ridge
ridge_reg = Ridge()
ridge_reg.fit(X, y, sample_weight=weights)

# get the R^2 value
r_squared = ridge_reg.score(X, y, sample_weight=weights)
print('R^2 on the training data:')
print(r_squared)

# get the coefficients for each feature
coef_data = list(zip(X.columns,ridge_reg.coef_))
coef_df = pd.DataFrame(coef_data,columns=['feature','feature_coef'])
coef_df.head()

# first get rid of opponent_ variables
team_df = coef_df[~coef_df['feature'].str.contains("opponent")]

# get rid of the location variable
team_df = team_df.iloc[1:]

team_df.head()

# rank them by coef, not alphabetical order
ranked_team_df = team_df.sort_values(['feature_coef'],ascending=False)
# reset the indices at 0
ranked_team_df = ranked_team_df.reset_index(drop=True);

ranked_team_df.head()

ranked_team_df.rename(columns={'feature':'team', 'feature_coef':'YUSAG_coef'}, inplace=True)
ranked_team_df['team'] = ranked_team_df['team'].str.replace('team_', '')

ranked_team_df.head()

ranked_team_df.index = ranked_team_df.index + 1 

ranked_team_df.to_csv("FCS_power_rankings.csv")

ivy_team_names = ['Yale','Harvard','Princeton','Cornell','Brown','Columbia','Dartmouth','Penn']
ivy_ranked_team_df = ranked_team_df[ranked_team_df['team'].isin(ivy_team_names)]

ivy_ranked_team_df

# sort by date and reset the indices to 0
df_dated = df.sort_values(['year', 'month','day'], ascending=[True, True, True])
df_dated = df_dated.reset_index(drop=True)

df_dated.head()

thirteen_df = df_dated.loc[df_dated['year']==2013]
fourteen_df = df_dated.loc[df_dated['year']==2014]
fifteen_df = df_dated.loc[df_dated['year']==2015]

train_df = pd.concat([thirteen_df,fourteen_df,fifteen_df], ignore_index=True)

sixteen_df = df_dated.loc[df_dated['year']==2016]
seventeen_df = df_dated.loc[df_dated['year']==2017]

test_df = pd.concat([sixteen_df,seventeen_df], ignore_index=True)

def train_test_model(train_df, test_df):

    # make the training data
    X_train = train_df.drop(['year','month','day','team','opponent','teamscore','oppscore','D1','OT','weights','scorediff'], axis=1)
    y_train = train_df['scorediff']
    weights_train = train_df['weights']
    
    # train the model
    ridge_reg = Ridge()
    ridge_reg.fit(X_train, y_train, weights_train)
    fit = ridge_reg.score(X_train,y_train,sample_weight=weights_train)
    print('R^2 on the training data:')
    print(fit)
    
    # get the test data
    X_test = test_df.drop(['year','month','day','team','opponent','teamscore','oppscore','D1','OT','weights','scorediff'], axis=1)
    y_test = test_df['scorediff']
    
    # get the metrics
    compare_data = list(zip(ridge_reg.predict(X_test),y_test))
    
    right_count = 0
    for tpl in compare_data:
        if tpl[0] >= 0 and tpl[1] >=0:
            right_count = right_count + 1
        elif tpl[0] <= 0 and tpl[1] <=0:
            right_count = right_count + 1
    accuracy = right_count/len(compare_data)
    print('accuracy on this weeks games')
    print(right_count/len(compare_data))
    
    total_squared_error = 0.0
    for tpl in compare_data:
        total_squared_error = total_squared_error + (tpl[0]-tpl[1])**2
    RMSE = (total_squared_error / float(len(compare_data)))**(0.5)
    print('RMSE on this weeks games:')
    print(RMSE)
    
    return fit, accuracy, RMSE, right_count, total_squared_error
     

#Now the code for running the week by week testing.
base_df = train_df
new_indices = []
# this is the hash for the first date
last_date_hash = 2026

fit_list = []
accuracy_list = []
RMSE_list = []
total_squared_error = 0
total_right_count = 0

for index, row in test_df.iterrows():
    
    year = row['year']
    month = row['month']
    day = row['day']
    date_hash = year+month+day 
    
    if date_hash != last_date_hash:
        last_date_hash = date_hash
        test_week = test_df.iloc[new_indices]
        fit, accuracy, RMSE, correct_calls, squared_error = train_test_model(base_df,test_week)
        
        fit_list.append(fit)
        accuracy_list.append(accuracy)
        RMSE_list.append(RMSE)
        
        total_squared_error = total_squared_error + squared_error
        total_right_count = total_right_count + correct_calls
        
        base_df = pd.concat([base_df,test_week],ignore_index=True)
        new_indices = [index]
        
    else:
        new_indices.append(index)

# get the number of games it called correctly in 2016
total_accuracy = total_right_count/test_df.shape[0]
total_accuracy

# get the Root Mean Squared Error
overall_RMSE = (total_squared_error/test_df.shape[0])**(0.5)
overall_RMSE

df.head()

# make the training data
X = df.drop(['year','month','day','team','opponent','teamscore','oppscore','D1','OT','weights','scorediff'], axis=1)
y = df['scorediff']
weights = df['weights']

# train the model so that we can get data on predicted score differentials
from sklearn.linear_model import Ridge
ridge_reg = Ridge()
ridge_reg.fit(X, y, sample_weight=weights)

df['predicted_scorediff'] = ridge_reg.predict(X)

df['predicted_scorediff'] = df['predicted_scorediff'].apply(lambda x: round(x,1))

last_year_df = df[df['year']==2016]

# ignore warnings from pandas (sorry my code probably isn't idiomatic)
import warnings
warnings.filterwarnings('ignore')

last_year_df.loc[last_year_df.scorediff >= 0, 'win'] = 1
last_year_df.loc[last_year_df.scorediff < 0, 'win'] = 0

last_year_df[['scorediff','predicted_scorediff','win']].head()

# make the training data
X = last_year_df['predicted_scorediff']
X = X.values.reshape(-1,1)
y = last_year_df['win']

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

last_year_df['winprob'] = 0
last_year_df['winprob'] = (1-log_reg.predict_proba(X))

last_year_df[['scorediff','predicted_scorediff','win','winprob']].head()

ivy_games_df = pd.read_csv('2017_IVY_schedule.csv')

ivy_games_df.head()

# make the rankings into a dictionary
rankings_dict = ranked_team_df.set_index('team')['YUSAG_coef'].to_dict()

# Initialize new columns
ivy_games_df['predicted_scorediff'] = 0
ivy_games_df['winprob'] = 0

ivy_games_df['team'] = ivy_games_df['team'].apply(lambda x: x.strip())
#df['team'] = df['team'].str.strip()
ivy_games_df['opponent'] = ivy_games_df['opponent'].apply(lambda x: x.strip())
#df['opponent'] = df['opponent'].str.strip()

# change 'Pennsylvania' to 'Penn'
ivy_games_df.loc[ivy_games_df.team == 'Pennsylvania','team'] = 'Penn'
ivy_games_df.loc[ivy_games_df.opponent == 'Pennsylvania','opponent'] = 'Penn'

# not working on Central Connecticut for some reason
# ivy_games_df.loc[ivy_games_df.opponent == 'Central Connecticut','team'] = 'Central Conn. St.'

ivy_games_df.head()

location_const = 1.77
for index, row in ivy_games_df.iterrows():
    
    # use a try in case team is unknown
    try:
        team_YUSAG_coef = rankings_dict[row['team']]
        opponent_YUSAG_coef = rankings_dict[row['opponent']]
    except:
        continue
        
    if row['Location'] == 1:
        ivy_games_df.loc[index,'predicted_scorediff'] = team_YUSAG_coef - opponent_YUSAG_coef + location_const
    elif row['Location'] == 0:
        ivy_games_df.loc[index,'predicted_scorediff'] = team_YUSAG_coef - opponent_YUSAG_coef

ivy_games_df.head()

predicted_scorediff_data = ivy_games_df['predicted_scorediff'].values.reshape(-1,1)
ivy_games_df['winprob'] = (1-log_reg.predict_proba(predicted_scorediff_data))

ivy_games_df.head()

conference_games_df = ivy_games_df.loc[ivy_games_df['team'].isin(ivy_team_names) & ivy_games_df['opponent'].isin(ivy_team_names)]

conference_games_df.head()

# create an initial dictionary to hold win records
ivy_records_dict = {}
# initialize each value to 0
for team in ivy_team_names:
    ivy_records_dict[team] = 0
# fill the entries with expected number of wins
for index, row in conference_games_df.iterrows():
    team = row['team']
    opponent = row['opponent']
    team_winprob = row['winprob']
    opponent_winprob = 1 - row['winprob']
    
    ivy_records_dict[team] = ivy_records_dict[team] + team_winprob
    ivy_records_dict[opponent] = ivy_records_dict[opponent] + opponent_winprob

ivy_records_dict # expected record is just (value, 7-value)

import random

# initialize a dict of league titile counts
ivy_titles_dict = {}
for team in ivy_team_names:
    ivy_titles_dict[team] = 0
    
# run through the season 10,000 times
for i in range(10000):
    # can change these values based on current record
    season_win_dict = dict({'Yale': 0, 'Harvard': 0,'Penn': 0,'Dartmouth': 0,
                            'Princeton': 0, 'Brown': 0,'Columbia': 0,'Cornell': 0,})
    # loop over the games in a season
    for index, row in conference_games_df.iterrows():
        rand_num = random.random()
        if rand_num <= row['winprob']:
            season_win_dict[row['team']] = season_win_dict[row['team']] + 1
        elif rand_num > row['winprob']:
            season_win_dict[row['opponent']] = season_win_dict[row['opponent']] + 1
    # find the largest number of wins
    most_wins = 0
    for key in season_win_dict:
        if season_win_dict[key] > most_wins:
            most_wins = season_win_dict[key]
    # find teams with the most wins, they get the title according to ivy rules
    for key in season_win_dict:
        if season_win_dict[key] == most_wins:      
            ivy_titles_dict[key] = ivy_titles_dict[key] + 1
            
    if i%1000 == 0:
        print(i)

ivy_titles_dict

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

plt.figure(figsize=(12,6))

labels = list(ivy_titles_dict.keys())
x_pos = np.arange(len(labels))
heights = [x/10000.0 for x in list(ivy_titles_dict.values())]

plt.bar(x_pos, heights, align='center')
plt.xticks(x_pos, labels)
plt.ylabel('Chance of winning title')
plt.title('Ivy League Season Simulations')
 
plt.show()

ivy_ranked_team_df

# add record and title chances to df
ivy_ranked_team_df['expected_wins'] = 0
ivy_ranked_team_df['expected_losses'] = 0
ivy_ranked_team_df['title_chances'] = 0
for index, row in ivy_ranked_team_df.iterrows():
    team = row['team']
    ivy_ranked_team_df.loc[index,'expected_wins'] = ivy_records_dict[team]
    ivy_ranked_team_df.loc[index,'expected_losses'] = 7 - ivy_records_dict[team]
    ivy_ranked_team_df.loc[index,'title_chances'] = ivy_titles_dict[team]/10000.0

ivy_ranked_team_df

ivy_ranked_team_df.to_csv("IVY_power_rankings.csv")

