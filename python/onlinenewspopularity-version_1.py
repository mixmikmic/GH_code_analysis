get_ipython().run_cell_magic('HTML', '', '<iframe width="640" height="360" src="https://mashable.com/" frameborder="0" gesture="media" allowfullscreen></iframe>')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

import seaborn as sns

input_file =  "C:/Users/jangn/OneDrive/CODE/Datasets/OnlineNewsPopularity/OnlineNewsPopularity.csv"
df = pd.read_csv(input_file, header = 0)

df.head(5)

df.shape

df.info()

df.describe() 

df.hist(bins=50, figsize= (16,16))
plt.show

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] =16.0
fig_size[1] = 4.0
#plt.rcParams["figure.figsize"] = fig_size

x = df['shares']
plt.hist(x, normed=True, bins=250)
plt.ylabel('shares');

def reject_outliers(shares):
    u = np.median(df['shares'])
    s = np.std(df['shares'])
    filtered= [e for e in (df['shares']) if (u - 2 * s < e < u + 2 * s)]
    return filtered

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] =16.0
fig_size[1] = 4.0
#plt.rcParams["figure.figsize"] = fig_size

filtered = reject_outliers('shares')
plt.hist(filtered, 100)
fig_size[0]=16.0
fig_size[1]=8.0
plt.show()

df_shares = pd.DataFrame(filtered)
df_shares.shape

df2 = df[df['shares']<26647]
df2.shape

corrmat = df2.corr()
corrmat['shares'].sort_values(ascending=False)

f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corrmat, vmax=1, square=True);
plt.show()

attributes = ['kw_avg_avg','LDA_03','kw_max_avg','kw_min_avg','num_hrefs',"shares"]
scatter_matrix(df2[attributes], figsize=(16, 16));  

attributes = ['num_imgs','self_reference_avg_sharess','is_weekend','self_reference_min_shares','self_reference_max_shares',"shares"]
scatter_matrix(df2[attributes], figsize=(16, 16));

attributes = ['kw_avg_max','global_subjectivity','abs_title_sentiment_polarity','weekday_is_sunday','title_subjectivity',"shares"]
scatter_matrix(df2[attributes], figsize=(16,16));

train_set, test_set = train_test_split(df2, test_size=0.20, random_state=42)

X_train_set = train_set.drop(['url','shares'], axis=1) #Dropping both 'shares', the predicted variable and 'url', a text variable
y_train_set = train_set['shares']

X_test_set = test_set.drop(['url','shares'], axis=1)
y_test_set = test_set['shares']

lin_reg = LinearRegression()
lin_reg.fit(X_train_set, y_train_set)

some_X_data = X_train_set.iloc[:500]
some_y_data = y_train_set.iloc[:500]
#print("Predicted shares:", lin_reg.predict(some_X_data))
#print("Actual shares:", list(some_y_data))

#We test how the model works by creating a dataframe from the sample. The df is then used as source for the seaborn plot below
df_someXdata = pd.DataFrame(lin_reg.predict(some_X_data),list(some_y_data) )
df_someXdata.reset_index(level=0, inplace=True)
df_someXdata_LR = df_someXdata.rename(index=str, columns={"index": "Actual shares", 0: "Predicted shares"})
df_someXdata_LR.head()

f, ax = plt.subplots(figsize=(17, 3))
sns.regplot(x=df_someXdata_LR["Actual shares"], y=df_someXdata_LR["Predicted shares"])
sns.plt.show()

share_predictions = lin_reg.predict(X_train_set)
lin_mse = mean_squared_error(y_train_set, share_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse #with outliers: 11648.966

df['shares'].median() #with outliers: 1400

lin_mae = mean_absolute_error(y_train_set, share_predictions)
print(lin_mae) 

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train_set, y_train_set)

some_X_data = X_train_set.iloc[:5]
some_y_data = y_train_set.iloc[:5]
print("Predicted shares:", tree_reg.predict(some_X_data))
print("Actual shares:", list(some_y_data))

share_predictions = tree_reg.predict(X_train_set)
tree_mse = mean_squared_error(y_train_set, share_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse 

tree_mae = mean_absolute_error(y_train_set, share_predictions)
print(tree_mae)

scores = cross_val_score(tree_reg, X_train_set, y_train_set,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores) 

lin_scores = cross_val_score(lin_reg, X_train_set, y_train_set,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores) #with outliers:mean 13185 std 7605

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train_set, y_train_set)

some_X_data = X_train_set.iloc[:500]
some_y_data = y_train_set.iloc[:500]
#print("Predicted shares:", lin_reg.predict(some_X_data))
#print("Actual shares:", list(some_y_data))

#We test how the model works by creating a dataframe from the sample. The df is then used as source for the seaborn plot below
df_someXdata = pd.DataFrame(forest_reg.predict(some_X_data),list(some_y_data) )
df_someXdata.reset_index(level=0, inplace=True)
df_someXdata_LR = df_someXdata.rename(index=str, columns={"index": "Actual shares", 0: "Predicted shares"})
df_someXdata_LR.head()

f, ax = plt.subplots(figsize=(17, 3))
sns.regplot(x=df_someXdata_LR["Actual shares"], y=df_someXdata_LR["Predicted shares"])
sns.plt.show()

share_predictions = forest_reg.predict(X_train_set)
forest_mse = mean_squared_error(y_train_set, share_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse #with outliers:5282.35

forest_mae = mean_absolute_error(y_train_set, share_predictions)
print(forest_mae) #with outliers:1424

scores = cross_val_score(forest_reg, X_train_set, y_train_set,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(forest_rmse_scores) 

param_grid = [
     {'n_estimators': [3,10], 'max_features':[2,3,4]},
    {'bootstrap': [False], 'n_estimators': [3,10],'max_features': [2,3,4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring="neg_mean_squared_error")

grid_search.fit(X_train_set, y_train_set)

cvres=grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

print(grid_search.best_params_)

param_distribs = {
        'n_estimators': randint(low=1, high=20),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train_set, y_train_set)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

attributes_all = ['kw_avg_avg','LDA_03','kw_max_avg','kw_min_avg','num_hrefs','num_imgs','self_reference_avg_sharess','is_weekend','self_reference_min_shares','self_reference_max_shares','kw_avg_max','global_subjectivity','abs_title_sentiment_polarity','weekday_is_sunday','title_subjectivity','data_channel_is_socmed','num_keywords','weekday_is_saturday','title_sentiment_polarity','num_videos','kw_avg_min','kw_max_min','data_channel_is_lifestyle','avg_positive_polarity','timedelta','global_sentiment_polarity','max_positive_polarity','data_channel_is_tech','global_rate_positive_words','kw_min_max','num_self_hrefs','LDA_04','kw_min_min','kw_max_max','global_rate_negative_words','n_tokens_content','n_unique_tokens','n_non_stop_words','n_non_stop_unique_tokens','min_positive_polarity','abs_title_subjectivity','weekday_is_friday','weekday_is_monday','LDA_00','max_negative_polarity','n_tokens_title','weekday_is_thursday','rate_positive_words','weekday_is_tuesday','min_negative_polarity','weekday_is_wednesday','LDA_01','rate_negative_words','avg_negative_polarity','data_channel_is_entertainment','data_channel_is_bus','average_token_length','data_channel_is_world','LDA_02']

sorted(zip(feature_importances, attributes_all), reverse=True)

df_nn = pd.DataFrame(feature_importances, attributes_all)
df_nn.reset_index(level=0, inplace=True)
df_nn.sort_values(0).rename(index=str, columns={"index": "Feature", 0: "importance"})

f, ax = plt.subplots(figsize=(16, 14))
sns.set_color_codes("pastel")
ax = sns.barplot( y='index', x= 0, data=df_nn.sort_values([0], ascending=[False]))

final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test_set)

final_mse = mean_squared_error(y_test_set, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

some_X_data = X_test_set.iloc[:] #seems to be working also w-o .iloc!
some_y_data = y_test_set.iloc[:] #seems to be working also w-o .iloc!

Predicted_shares = list(final_model.predict(some_X_data)) 

Actual_shares = list(some_y_data)

final_data = [Predicted_shares, Actual_shares]

sorted(zip(Predicted_shares, Actual_shares), reverse=True);

df_shares = pd.DataFrame(Predicted_shares, Actual_shares)   
df_shares.reset_index(level=0, inplace=True)
df_shares_AvsP = df_shares.rename(index=str, columns={"index": "Actual shares", 0: "Predicted shares"})
df_shares_AvsP.head()

f, ax = plt.subplots(figsize=(17, 6))
sns.regplot(x=df_shares_AvsP["Actual shares"], y=df_shares_AvsP["Predicted shares"])
sns.plt.show()

fig_size[0]=17.0
fig_size[1]=6.0

df_shares_AvsP.plot(bins=300, kind='hist', alpha=0.7)
plt.title('Number of actual shares of online news vs. number of prediced shares by the Random Forest model')
axes=plt.axes()
plt.show()

from sklearn.metrics import r2_score

r2_score( Actual_shares, Predicted_shares)

