##
# The 'import' statement imports external libraries for use in the interactive session.
# ... and 'import <library> as <nickname>' makes a shorter name for convenience.
#
# The '%matplotlib inline' statement allows inline plots here. (see try.jupyter.org)
#
import datetime
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Bigger fonts and figures for the demo
matplotlib.rcParams.update({
        'font.size': 14,
        'figure.figsize':(10.0, 8.0),
        'axes.formatter.useoffset':False })

# Better data frame display for the demo
pd.set_option('expand_frame_repr', True)
pd.set_option('max_rows', 18)
pd.set_option('max_colwidth', 14)
pd.set_option('precision',2)

# Evaluate this cell for documentation on Jupyter magics
get_ipython().magic('magic')

##
# Pro tip:
#    You probably know 'help(object)' will show documentation for the object.
#    'dir(object)' will list all names accessible via the object.
print(' '.join([x for x in dir(pd) if 'read_' in x]))

## ... and output options:
print(' '.join([x for x in dir(pd.DataFrame) if 'to_' in x]))

import os  # in case you're on windows
file_location = os.path.join('data', 'nfl_season2008to2014.csv')
df = pd.read_csv(file_location)

df.head(3)

##
# Transformations using Pandas

# Spread
#    The pandas.DataFrame uses numpy arrays to enable array-wise operations.
df['Spread'] = df.Points - df.PointsAllowed

# PtsPct
df['PointsPercent'] = df.Points / (df.Points + df.PointsAllowed)

# Outcome
#    When assigning to a subset of a column in a DataFrame,
#    use indexing ('ix' or other functions) to identify the subset.
df['Outcome'] = np.nan  # pre-fill the column
df.ix[df.Spread > 0, 'Outcome'] = 'W'
df.ix[df.Spread < 0, 'Outcome'] = 'L'
df.ix[df.Spread == 0, 'Outcome'] = 'T'

# WLT (record)
#    Make sure the data are sorted.
#    Then:
#    Use 'apply' for user-defined functions
#    and 'cumsum' for a running sum; rows with np.nan
#    in the outcome will just add zero
df = df.sort(['Team','Season','Week'])
df['WLT'] = df.groupby(['Team','Season'])['Outcome'].apply(
    lambda o:
        (o == 'W').cumsum().astype(str) + '-' +
        (o == 'L').cumsum().astype(str) + '-'+
        (o == 'T').cumsum().astype(str) )

# WinPct
df['WinPct'] = df.groupby(('Team','Season'))['Spread'].apply(
    lambda s: (0.5 * (s == 0).cumsum() + (s > 0).cumsum()) / s.notnull().cumsum() )

# LastWkBye
#    Make sure the data are sorted.
#    Then flag whether last game was a Bye
df = df.sort(['Team', 'Season','Week'])
df['LastWeekBye'] = df.groupby(['Team','Season'])['Spread'].shift(1).isnull().fillna(False)
df.ix[df.Week == 1, 'LastWeekBye'] = False

# Past5WkAvgPts
#    Make sure the data are sorted.
#    Then use the windowing functions
#    see: http://pandas.pydata.org/pandas-docs/stable/computation.html
#    window size = 5, minimum required observations = 2
df = df.sort(['Team','Season', 'Week'])
df['Past5WkAvgPts'] = df.groupby(['Team']).Points.apply(
    pd.rolling_mean, window=5, min_periods=2).shift(1)

# Past5WkAvgFumbles
#    Some of the sorting seems unnecessary
#    but is good in case you copy and paste a snippet elsewhere...
df = df.sort(['Team','Season','Week'])
df['Past5WkAvgFumbles'] = df.groupby(['Team']).Fumbles.apply(
    pd.rolling_mean, window=5, min_periods=2).shift(1)

# Past5WkInterceptions
df = df.sort(['Team','Season','Week'])
df['Past5WkInterceptions'] = df.groupby('Team').Interceptions.apply(
    pd.rolling_sum, window=5, min_periods=2).shift(1)

# EwmaPenaltyYards, centered at 2 Weeks ago
#  exponentially weighted moving average
df = df.sort(['Team','Season','Week'])
df['EwmaPenaltyYards'] = df.groupby('Team').PenaltyYards.apply(
    pd.ewma, 2).shift(1)


df.head(9)

##
# Distribution of the point spread
#
plt.subplot(221)
df.Spread.plot(kind='density')
plt.title('Spread 2008-2014')
plt.xlim((-60,60))
plt.gcf().set_size_inches(12, 5)
    
plt.subplot(222)
df.Spread.hist(bins=60)
plt.title('...but not many ties')

plt.subplot(212)
plt.plot(df.Spread, df.index, 'ko', alpha=0.1)
plt.xlabel('All games')

plt.show()

##
# Spread distribution by year
#   -- demonstrates pandas.DataFrame.pivot_table
#
df[['Week', 'Team', 'Season','Spread']].pivot_table(
    index=['Week', 'Team'], columns='Season', values='Spread').hist(
    bins=25, layout=(2,4), figsize=(12,3.5), sharey=True)
plt.show()

##
# Or cumulative distribution of spread
#    Uses the same 'hist' function as above but with different keyword arguments.
#
df.Spread.hist(cumulative=True, normed=True, histtype='step', bins=100, figsize=(9,4))
#- The shape is consistent season-over-season. If you want to see that uncomment the below....
#df.groupby('Season').Spread.hist(cumulative=True, normed=True, histtype='step', bins=100, figsize=(9,4))

plt.title('Cumulative Spread Distribution, NFL 2008-2014')
plt.ylim(0,1)
plt.xlabel('Spread')
plt.ylabel('Percent of Teams below Spread')
plt.axvline(x=7, hold=None, color='lightblue', linewidth=3)
plt.axvline(x=14, hold=None, color='lightblue', linewidth=2)
plt.axvline(x=21, hold=None, color='lightblue', linewidth=1)
plt.show()

##
# We can do things like explore whether home field advantage exists.
#
df[['AtHome', 'Spread']].boxplot(by='AtHome', figsize=(7,4))
plt.suptitle('')
plt.title('Difference in spread for Home vs Away teams')
plt.ylabel('Spread'); plt.xlabel('Playing at Home?')
plt.show()

##
# ... by Win counts now
fig = plt.figure()
ax = plt.subplot(111)

pd.crosstab(df.AtHome, df.Outcome).plot(
    ax=ax, kind='barh', stacked=True,
    color=['red','blue', 'green'], figsize=(9,5))

# Mom says label your axes
plt.ylabel('At Home?'); plt.xlabel('Number of Outcomes')

# Shrink current axis by 20% to make room for the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
plt.show()

##
# ... or see which teams are consistently great or not
#
df2 = pd.DataFrame({col:vals.Spread for col,vals in df.groupby('Team')})
meds = df2.median().order(ascending=True)
df2[meds.index].boxplot(vert=False, figsize=(5,35), return_type='axes',fontsize=14)
plt.xlabel('Spreads in Seasons 2008-2014')
plt.show()

##
# ... Some teams make points but not wins
#
df2 = pd.DataFrame({col:vals.Points for col,vals in df[df.Season==2014].groupby('Team')})
meds = df2.median().order(ascending=True)
df2[meds.index].boxplot(vert=False, figsize=(5,35), return_type='axes',fontsize=14)
plt.xlabel('Distribution of points')
plt.title("Points in the 2014 Season")
plt.show()

##
# ... How about Sacks?
#
df[(df.Season==2014) & (df.Category == 'regular')
  ].groupby('Team').Sacks.sum().order().plot(kind='barh')
plt.title("Sacks made in the 2014 season")
plt.show()

fig = plt.figure()
ax = plt.subplot(111)

tmp = df[df.Team.isin(['Chicago Bears','Detroit Lions','Denver Broncos'])
        ].groupby(['Team', 'Season']
           ).Spread.median().unstack().transpose()
tmp.plot(ax=ax, lw=3, figsize=(12,4), legend=False)

plt.xlabel('Season')
plt.ylabel('Median Spread')

# Shrink current axis by 20% to make room for the annotation
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Legend to the right of the current axis
ax.annotate('... Manning?',(2012, tmp.ix[2012,'Denver Broncos']), xytext=(2010.2, 10.5), 
             arrowprops=dict(arrowstyle='->'))
ax.annotate('... Suh?',(2010, tmp.ix[2012,'Detroit Lions']), xytext=(2010, -11), 
             arrowprops=dict(arrowstyle='->'))

for y, txt in zip([-7,7,2],
                  ['Chicago Bears', 'Denver Broncos', 'Detroit Lions']):
    ax.annotate(txt, (2014, y), xytext=(2014.1, y))

plt.title("It's a team sport, but...")
plt.show()

df.columns

##
# Combine the data by game, so that the opponent aggregations are available too
tm = df[[
        'Season', 'Category', 'Week', 'Team', 'Opponent', 'Spread', 'AtHome',
        'LastWeekBye',
        'Past5WkAvgPts',
        'Past5WkAvgFumbles', 'Past5WkInterceptions',
        'EwmaPenaltyYards']]
tm.columns = [
        'Season', 'Category', 'Week', 'Team', 'Opponent', 'Spread', 'AtHome',
        'T_LastWkBye',
        'T_5WkAvgPts',
        'T_5WkAvgFumbles', 'T_5WkAvgInterceptions',
        'T_EwmaPenaltyYards']

opp = df[[
        'Season', 'Category', 'Week', 'Team', 'Opponent',
        'LastWeekBye',
        'Past5WkAvgPts',
        'Past5WkAvgFumbles', 'Past5WkInterceptions',
        'EwmaPenaltyYards']]

opp.columns = [
        'Season', 'Category', 'Week', 'Opponent', 'Team',
        'O_LastWkBye',
        'O_5WkAvgPts',
        'O_5WkAvgFumbles', 'O_5WkAvgInterceptions',
        'O_EwmaPenaltyYards']

games = tm.merge(opp, how='inner', on=['Season', 'Category', 'Week', 'Team', 'Opponent'])

games = games[games.Spread.notnull()]
print('games shape:', games.shape)
print('df shape:', df.shape, 'df no bye', df[df.Points.notnull()].shape)

##
# All of the Scikit-learn models are trained with
#   (1) an output column  and
#   (2) an input dataset
#
# Want to predict 'Spread' given known values:
#   ==> Ignore 'Bye' weeks (they have no 'Spread')
#   - Category (regular|postseason)
#   - AtHome (True|False)
#   - LastWeekBye (True|False)
#   - Past5WeekAveragePointsPercent (Numeric)
#   - LastSeasonPlayoffGames (Numeric-count)
#   - LastSeasonPointsPercent (Numeric)
#
no_nulls = games.notnull().all(axis=1)
spread = games[no_nulls].Spread
input_data = games[no_nulls][[
        'Team', # 'Opponent',
        'Category',
        'AtHome',
        'T_LastWkBye',
        'T_5WkAvgPts', 'T_5WkAvgFumbles', 'T_5WkAvgInterceptions', 'T_EwmaPenaltyYards',
        'O_LastWkBye',
        'O_5WkAvgPts', 'O_5WkAvgFumbles', 'O_5WkAvgInterceptions', 'O_EwmaPenaltyYards'
    ]]

# The input column 'Category' contains categories,
# so we have to make dummy variables to use in the regression.
input_data = pd.get_dummies(input_data)

print("Size of the input set:", input_data.shape)
input_data.head(3)

fig, axs = plt.subplots(2, 5)
fig.set_size_inches(14, 6)

cols = ('T_LastWkBye', 'T_5WkAvgPts', 'T_5WkAvgFumbles', 'T_5WkAvgInterceptions', 'T_EwmaPenaltyYards',
        'O_LastWkBye', 'O_5WkAvgPts', 'O_5WkAvgFumbles', 'O_5WkAvgInterceptions', 'O_EwmaPenaltyYards')

for ax, col in zip(axs.flatten(), cols):
    ax.scatter(y=games.Spread, x=games[col], alpha=0.1)
    ax.set_title(col, fontsize=12)

plt.show()

# Set up cross-validation
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

def perform_kfold_cross_validation(model, all_X, all_y, k=5):
    """Calculate root mean squared error for each cross-validation fold.
    
    Parameters:
        model - a scikit learn model
        all_X - a pandas DataFrame with the observed input data
        all_y - a pandas Series with the observed outcome
        k - number of cross validation folds (test set will be 1/k of the data)
    
    Return value:
        An array of length 'k' with the root mean squared error
        for each fold.
    """
    # 'folds' is a generator that will yield pairs of arrays (train, test)
    # selecting row numbers for training/testing
    folds = cross_validation.KFold(n=len(all_y), n_folds=k)
    RMSE = []    # root mean squared errors
    # Loop over the cross-validation folds
    for training, testing in folds:
        # Get the training and test splits
        training = all_X.index[training]
        testing = all_X.index[testing]
        X_train, X_test = all_X.ix[training], all_X.ix[testing]
        y_train, y_test = all_y.ix[training], all_y.ix[testing]
    
        # Train the model
        model.fit(X_train, y_train)
        # Use the model to predict output
        y_fitted = model.predict(X_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_fitted)))
    # Leave the model fit to the entire dataset
    model.fit(all_X, all_y)
    # And return the array of root mean squared errors
    return RMSE

##
# Some popular regression models
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import tree

# To see available tuning settings are for a model,
# call help on its initialization function, e.g. :
#  help(linear_model.Ridge.__init__)
models = dict(
    ols = linear_model.LinearRegression(),
    gbm = ensemble.GradientBoostingRegressor(max_depth=5),
    ridge = linear_model.Ridge(),
    svr = svm.LinearSVR(epsilon=2),
    tree = tree.DecisionTreeRegressor(max_depth=5),
    random_forest = ensemble.RandomForestRegressor(n_estimators=5, max_depth=5)
)

rmses = {}
for name, model in models.items():
    rmses[name] = perform_kfold_cross_validation(model, input_data, spread, k=8)
    
pd.DataFrame(rmses).boxplot(vert=False, return_type='axes')
plt.gcf().set_size_inches(9, 5)
plt.xlabel("Error in predicted spread"); plt.ylabel("Model")
plt.show()

fig, axs = plt.subplots(2, 3, sharey=True)
fig.set_size_inches(12, 8)

# Make the train/test split be pre-2014/2014
train = games[no_nulls].Season < 2014
test = games[no_nulls].Season == 2014

for (ax, (name, model)) in zip(axs.flatten(), models.items()):
    model.fit(input_data.ix[train], spread[train])
    ax.scatter(x=spread[test], y=model.predict(input_data.ix[test]), alpha=0.2)
    ax.plot((-60,60), (-60,60), ls="--", c=".3", color='gray')  # Diagonal line 1=1
    ax.set_title(name)
    ax.set_ylim(-30,30)
    ax.set_ylabel('Predicted')
    
plt.show()

# To see available tuning settings are for a model,
# call help on its initialization function, e.g. :
#  help(linear_model.Ridge.__init__)
from sklearn import naive_bayes

# We can use the Naive Bayes classifier so long as there are no
# columns with negative values
input_data = games[no_nulls][[
        'Team', 'Opponent',
        'AtHome',
        'T_LastWkBye',
        'T_5WkAvgPts', 'T_5WkAvgFumbles', 'T_5WkAvgInterceptions', 'T_EwmaPenaltyYards',
        'O_LastWkBye',
        'O_5WkAvgPts', 'O_5WkAvgFumbles', 'O_5WkAvgInterceptions', 'O_EwmaPenaltyYards'
    ]]

# The input columns 'Team' and 'Opponent', contain categories,
# so we have to make dummy variables to use in the regression.
input_data = pd.get_dummies(input_data)
print("Size of the input set:", input_data.shape)

models = dict(
    logistic = linear_model.LogisticRegression(),
    gbc = ensemble.GradientBoostingClassifier(max_depth=5),
    ridge = linear_model.RidgeClassifier(),
    tree = tree.DecisionTreeClassifier(max_depth=5),
    #svc = svm.LinearSVC(),
    naive_bayes = naive_bayes.MultinomialNB(),  # Can only use if all inputs are positive
    random_forest = ensemble.RandomForestClassifier(n_estimators=10, max_depth=5)
)

win = (spread > 0).astype(int)
rmses = {}
for name, model in models.items():
    rmses[name] = perform_kfold_cross_validation(model, input_data, win, k=3)
    
pd.DataFrame(rmses).boxplot(vert=False, return_type='axes')
plt.gcf().set_size_inches(9, 5)
plt.xlabel("Error in prediction"); plt.ylabel("Model")
plt.show()

fig, axs = plt.subplots(2, 3, sharey=True)
fig.set_size_inches(12, 8)

# Make the train/test split be pre-2014/2014
train = games[no_nulls].Season < 2014
test = games[no_nulls].Season == 2014

for (ax, (name, model)) in zip(axs.flatten(), models.items()):
    pd.crosstab(win.apply(lambda x: ('Lose', 'Win')[x]),
                pd.Series(('Pred. Lose', 'Pred. Win')[x] for x in model.predict(input_data))).plot(
    ax=ax, kind='barh', stacked=True, legend=None,
    color=['red','green'], figsize=(9,5))
    if ax.is_last_col():
        # Legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=False)
    ax.set_xticks([])
    ax.set_title(name)
    ax.set_ylabel('Actual')

plt.show()

##
# ROC charts
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
from sklearn.metrics import roc_curve, auc

fig, axs = plt.subplots(2, 3, sharey=True)
fig.set_size_inches(12, 8)

for (ax, (name, model)) in zip(axs.flatten(), models.items()):
    try:
        fpr, tpr, _ = roc_curve(win[test], model.predict_proba(input_data[test])[:,1])
    except:
        fpr, tpr, _ = roc_curve(win[test], model.decision_function(input_data[test]))
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label='ROC curve\n(area = {:0.2f})'.format(roc_auc))
    ax.plot((0,0), (1,1), ls="--", c=".3", color='lightgray')  # Diagonal line 1=1
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right', fontsize=12, frameon=False)
    if ax.is_first_col():
        ax.set_ylabel('True Positive Rate')
    ax.set_xticks([])
    ax.set_xlabel('False Positive Rate')
    ax.set_title(name)

plt.show()

# Week by week

# Pick the GBC model since that looked best
model = models['gbc']

# Pick columns to show the results
results = games[no_nulls][['Season', 'Week', 'Team', 'Opponent', 'AtHome']]

for wk in range(1, 18):
    # Each week we want only the dates before
    print wk, '...',
    train = (games[no_nulls].Season < 2014) | (games[no_nulls].Week < wk)
    test = (games[no_nulls].Season == 2014) & (games[no_nulls].Week == wk)
    model.fit(input_data[train], win[train])
    probability = model.predict_proba(input_data[test])[:,1]
    results.ix[test, 'Win_Actual'] = win[test]
    results.ix[test, 'Win_Predicted'] = probability

results.shape

results = results[(results.Season==2014) & (results.Week < 18)]

# Merge on the home team
resultsH = results[results.AtHome]
resultsA = results[~results.AtHome]
del resultsH['AtHome']
del resultsA['AtHome']
del resultsA['Win_Actual']

resultsH.columns = ['Season', 'Week', 'Team', 'Opponent', 'Home_Win', 'Home_Pred_W']
resultsA.columns = ['Season', 'Week', 'Opponent', 'Team', 'Away_Pred_W']
resultsH = resultsH.merge(resultsA, on=['Season', 'Week', 'Team', 'Opponent'])

resultsH.columns

resultsH.sort(['Week', 'Team'])

resultsH['Odds'] = resultsH.Home_Pred_W / resultsH.Away_Pred_W
resultsH['Rank'] = resultsH.groupby('Week').Odds.rank()
resultsH = resultsH.sort(['Week','Rank'], ascending=False)
resultsH[resultsH.Week == 1]

results.ix[results.Season==2014, ['Win_Actual', 'Win_Predicted']].boxplot(
    by='Win_Actual', figsize=(7,4))
plt.suptitle('')
plt.title('Actual Wins vs Prediction')
plt.ylabel('Prediction'); plt.xlabel('Won?')
plt.show()

# The script below does just this
get_ipython().magic('run make_predictions.py')

# And the other code
get_ipython().magic('run extra_code/make_datasheet.py')

# ...Finally
get_ipython().magic('run extra_code/make_gamesheets.py')

