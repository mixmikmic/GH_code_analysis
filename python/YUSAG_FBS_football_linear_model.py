import numpy as np
import pandas as pd
import math

df_1 = pd.read_csv('NCAA_FBS_Results_2013_.csv')
df_2 = pd.read_csv('NCAA_FBS_Results_2014_.csv')
df_3 = pd.read_csv('NCAA_FBS_Results_2015_.csv')
df_4 = pd.read_csv('NCAA_FBS_Results_2016_.csv')

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

ranked_team_df.to_csv("FBS_power_rankings.csv")

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
last_date_hash = 2018

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

