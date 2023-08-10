import pandas as pd #for building dataframes from CSV files
import glob, os #for reading file names
import seaborn as sns #for fancy charts
import numpy as np #for np.nan
from scipy import stats #for statistical analysis
from scipy.stats import norm #for statistical analysis
from datetime import datetime #for time-series plots
import statsmodels #for integration with pandas and analysis
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')

#Initialize dataframes given API data
my_beer_df = pd.read_csv('../data/my-final-beer-data.csv')
my_new_beer_df = pd.read_csv('../data/final-avg-user-beer-data.csv')

my_beer_df.info()

my_new_beer_df.info()

#Select only certain columns for now
my_beer_df.columns

#Creating the dataframe for machine learning to test
new_beer_df = my_beer_df[['beer.beer_abv',
       'beer.beer_ibu', 
       'beer.beer_style', 'beer.bid', 'beer.rating_count',
       'beer.rating_score', 'brewery.brewery_active',
       'brewery.brewery_id', 'brewery.country_name',
       'brewery.location.brewery_city', 'brewery.location.brewery_state',]]

#Creating the dataset of untasted beers
untasted_beer_df = my_new_beer_df[['beer_abv',
       'beer_ibu', 
       'beer_style', 'bid', 'rating_count',
       'rating_score', 'is_in_production',
       'brewery.brewery_id', 'brewery.country_name',
       'brewery.location.brewery_city', 'brewery.location.brewery_state',]]

my_ratings = my_beer_df['beer.auth_rating']

new_beer_df.head()

new_beer_df.info()

untasted_beer_df.info()

#Need to create categorical variables for the necessary features
new_beer_df["beer.beer_style"] = new_beer_df["beer.beer_style"].astype('category')
new_beer_df["beer.beer_style_cat"] = new_beer_df["beer.beer_style"].cat.codes
new_beer_df["brewery.country_name"] = new_beer_df["brewery.country_name"].astype('category')
new_beer_df["brewery.country_name_cat"] = new_beer_df["brewery.country_name"].cat.codes
new_beer_df["brewery.location.brewery_city"] = new_beer_df["brewery.location.brewery_city"].astype('category')
new_beer_df["brewery.location.brewery_city_cat"] = new_beer_df["brewery.location.brewery_city"].cat.codes
new_beer_df["brewery.location.brewery_state "] = new_beer_df["brewery.location.brewery_state "].astype('category')
new_beer_df["brewery.location.brewery_state _cat"] = new_beer_df["brewery.location.brewery_state "].cat.codes

#Need to create categorical variables for the necessary features
untasted_beer_df["beer_style"] = untasted_beer_df["beer_style"].astype('category')
untasted_beer_df["beer_style_cat"] = untasted_beer_df["beer_style"].cat.codes
untasted_beer_df["brewery.country_name"] = untasted_beer_df["brewery.country_name"].astype('category')
untasted_beer_df["brewery.country_name_cat"] = untasted_beer_df["brewery.country_name"].cat.codes
untasted_beer_df["brewery.location.brewery_city"] = untasted_beer_df["brewery.location.brewery_city"].astype('category')
untasted_beer_df["brewery.location.brewery_city_cat"] = untasted_beer_df["brewery.location.brewery_city"].cat.codes
untasted_beer_df["brewery.location.brewery_state "] = untasted_beer_df["brewery.location.brewery_state "].astype('category')
untasted_beer_df["brewery.location.brewery_state _cat"] = untasted_beer_df["brewery.location.brewery_state "].cat.codes

new_beer_df.columns

untasted_beer_df.columns

#includes the avg user ratings for each beer
final_beer_df = new_beer_df[['beer.beer_abv', 'beer.beer_ibu', 'beer.bid',
       'beer.rating_count', 'beer.rating_score', 'brewery.brewery_active',
       'brewery.brewery_id',
       'beer.beer_style_cat', 'brewery.country_name_cat',
       'brewery.location.brewery_city_cat']]

# do NOT include avg user ratings
final_beer_df2 = new_beer_df[['beer.beer_abv', 'beer.beer_ibu',
        'brewery.brewery_active',
       'brewery.brewery_id',
       'beer.beer_style_cat', 'brewery.country_name_cat',
       'brewery.location.brewery_city_cat']]

final_untasted_beer_df = untasted_beer_df[['beer_abv', 'beer_ibu', 'bid',
       'rating_count', 'rating_score', 'is_in_production',
       'brewery.brewery_id',
       'beer_style_cat', 'brewery.country_name_cat',
       'brewery.location.brewery_city_cat']]

from sklearn.model_selection import train_test_split

y = my_ratings # define the target variable (dependent variable) as y

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(final_beer_df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

print('Score: ' + str(rf.score(X_test, y_test)))

from sklearn.grid_search import GridSearchCV
param_grid = {'oob_score': [True, False], 
                 'min_samples_split': [8, 16], 
                 'min_samples_leaf': [4, 8]}
rfc = GridSearchCV(RandomForestRegressor(), param_grid, cv=10)
rfc.fit(X_train, y_train)

print('Acc: ' + str(rfc.score(X_test, y_test)))

rf.predict(final_untasted_beer_df)

predictions_df = pd.DataFrame(rf.predict(final_untasted_beer_df), columns=['predicted_values'])

predictions_df.head()

final_predict_df = pd.concat([final_untasted_beer_df, predictions_df], axis=1)

final_predict_df.columns

final_predict_df.info()

final_predict_df.sort_values(by='predicted_values', ascending=False).head(5)

my_new_beer_df[my_new_beer_df['bid']==7936]

#list(zip(final_beer_df.columns, rf.feature_importances_))
plot_df = pd.DataFrame()
plot_df['col_names'] = final_beer_df.columns
plot_df['Predictions'] = rf.feature_importances_
plot_df.sort_values(by='Predictions', ascending=False)

plt.rc("figure", figsize=(12, 8))
g = sns.barplot(x = "col_names", y = "Predictions", data = plot_df.sort_values(by='Predictions', ascending=False))
g.set_xticklabels(labels=plot_df.sort_values(by='Predictions', ascending=False)['col_names'],rotation=90)

# create training and testing vars
X_train2, X_test2, y_train2, y_test2 = train_test_split(final_beer_df2, y, test_size=0.2)
print(X_train2.shape, y_train2.shape)
print(X_test2.shape, y_test2.shape)

from sklearn.ensemble import RandomForestRegressor
rf2 = RandomForestRegressor()
rf2.fit(X_train2, y_train2)

print('Score: ' + str(rf2.score(X_test2, y_test2)))

#list(zip(final_beer_df.columns, rf.feature_importances_))
plot_df2 = pd.DataFrame()
plot_df2['col_names'] = final_beer_df2.columns
plot_df2['Predictions'] = rf2.feature_importances_
plot_df2.sort_values(by='Predictions', ascending=False)

g2 = sns.barplot(x = "col_names", y = "Predictions", data = plot_df2.sort_values(by='Predictions', ascending=False))
g2.set_xticklabels(labels=plot_df2.sort_values(by='Predictions', ascending=False)['col_names'],rotation=90)

final_beer_df.info()

# create training and testing vars
X_train3, X_test3, y_train3, y_test3 = train_test_split(final_beer_df, y, test_size=0.2)
print(X_train3.shape, y_train3.shape)
print(X_test3.shape, y_test3.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train3, y_train3)

print('Score: ' + str(lr.score(X_test3, y_test3)))

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(lr, final_beer_df, y, cv=5)
cv_scores

print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

# create training and testing vars
X_train4, X_test4, y_train4, y_test4 = train_test_split(final_beer_df2, y, test_size=0.2)
print(X_train4.shape, y_train4.shape)
print(X_test4.shape, y_test4.shape)

from sklearn.linear_model import LinearRegression
lr2 = LinearRegression()
lr2.fit(X_train4, y_train4)

print('Score: ' + str(lr2.score(X_test4, y_test4)))

cv_scores2 = cross_val_score(lr2, final_beer_df2, y, cv=5)
cv_scores2

print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores2.mean(), cv_scores2.std() * 2))

from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

print('Score: ' + str(gb.score(X_test, y_test)))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, gb.predict(X_test))
print("MSE: %.4f" % mse)

#list(zip(final_beer_df.columns, rf.feature_importances_))
plot_df3 = pd.DataFrame()
plot_df3['col_names'] = final_beer_df.columns
plot_df3['Predictions'] = gb.feature_importances_
plot_df3.sort_values(by='Predictions', ascending=False)

g3 = sns.barplot(x = "col_names", y = "Predictions", data = plot_df3.sort_values(by='Predictions', ascending=False))
g3.set_xticklabels(labels=plot_df3.sort_values(by='Predictions', ascending=False)['col_names'],rotation=90)



gb2 = GradientBoostingRegressor()
gb2.fit(X_train2, y_train2)

print('Score: ' + str(gb2.score(X_test2, y_test2)))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test2, gb2.predict(X_test2))
print("MSE: %.4f" % mse)

#list(zip(final_beer_df.columns, rf.feature_importances_))
plot_df4 = pd.DataFrame()
plot_df4['col_names'] = final_beer_df2.columns
plot_df4['Predictions'] = gb2.feature_importances_
plot_df4.sort_values(by='Predictions', ascending=False)

g4 = sns.barplot(x = "col_names", y = "Predictions", data = plot_df4.sort_values(by='Predictions', ascending=False))
g4.set_xticklabels(labels=plot_df4.sort_values(by='Predictions', ascending=False)['col_names'],rotation=90)





