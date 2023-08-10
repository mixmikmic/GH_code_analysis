import json
import numpy as np
import pandas as pd

# load and preview the data
business_data = []
with open('data/business_sample_cleveland.json') as f:
    for line in f:
        business_data.append(json.loads(line))
business_df = pd.DataFrame.from_dict(business_data)
business_df.head()

# "name" column name is ambiguous with df.name - change it
business_df = business_df.rename(columns = {'name': 'BusinessName'})

business_df['categories_clean'] = map(lambda x: '|'.join(x), business_df['categories'])
categories_df = business_df.categories_clean.str.get_dummies(sep='|')
# merge
business_df = business_df.merge(categories_df, left_index=True, right_index=True)
# remove intermediate columns (no longer needed)
business_df.drop(['categories', 'categories_clean'], axis=1, inplace=True)
business_df.head()

business_df['attributes'].head()

business_df = business_df.join(pd.DataFrame(business_df['attributes'].to_dict()).T)
# further split sub-attributes into their own columns
cols_to_split = ['BusinessParking', 'Ambience', 'BestNights', 'GoodForMeal', 'HairSpecializesIn', 'Music']
for col_to_split in cols_to_split:
    new_df = pd.DataFrame(business_df[col_to_split].to_dict()).T
    new_df.columns = [col_to_split + '_' + str(col) for col in new_df.columns]
    business_df = business_df.join(new_df)

business_df.drop(['attributes'] + cols_to_split, axis=1, inplace=True)
business_df.head()

# columns with non-boolean categorical values:
cols_to_split = ['AgesAllowed', 'Alcohol', 'BYOBCorkage', 'NoiseLevel', 'RestaurantsAttire', 'Smoking', 'WiFi']
new_cat = pd.concat([pd.get_dummies(business_df[col], prefix=col, prefix_sep='_') for col in cols_to_split], axis=1)
# keep all columns (not n-1) because 0's for all of them indicates that the data was missing (useful info)
business_df = pd.concat([business_df, new_cat], axis=1)
business_df.drop(cols_to_split, inplace=True, axis=1)
business_df.head()

# convert true/false columns to 0/.5/1 for false/missing/true
print business_df['BusinessAcceptsCreditCards'].head(10)
business_df = business_df.fillna(0.5).apply(pd.to_numeric, errors='ignore')  # can narrow with .iloc[:,648:722] if necessary
business_df['BusinessAcceptsCreditCards'].head(10)

# deal with missing values in postal code
print business_df['postal_code'].isnull().sum()
business_df['postal_code'] = business_df['postal_code'].fillna(0)
print business_df['postal_code'].isnull().sum()

# check that all nulls are removed
business_df.isnull().sum().sum()

checkin_data = []
with open('data/checkin_sample_cleveland.json') as f:
    for line in f:
        checkin_data.append(json.loads(line))
checkin_df = pd.DataFrame.from_dict(checkin_data)
checkin_df.head()

# separate the values from the dict so they're a list of ['Day', 'time', count]
checkin_df['time_clean'] = map(lambda (x, y): map(lambda (k, v): map(lambda(i, value): [k, value, v.values()[i]], enumerate(v)), y.iteritems()), checkin_df['time'].iteritems())
# flatten the list so each day is no longer in its own list
checkin_df['time_clean'] = map(lambda l: [item for sublist in l for item in sublist], checkin_df['time_clean'])
# make it a cleaner dict where key='Day Time', value=count
checkin_df['time_clean'] = map(lambda x: {s + ' ' + t: u for (s, t, u) in x}, checkin_df['time_clean'])
# add column for each day/time with counts as values
new_checkin_df = checkin_df.join(pd.DataFrame(checkin_df["time_clean"].to_dict()).T).fillna(0)
# remove intermediate columns (no longer needed)
new_checkin_df.drop(['time', 'time_clean'], axis=1, inplace=True)
new_checkin_df.head()

business_df = business_df.merge(new_checkin_df, left_on='business_id', right_on='business_id', how='left')
business_df.head()

business_df.iloc[:, 648:] = business_df.iloc[:, 648:].fillna(0)

basic_cols = business_df.columns[:13]
print basic_cols

category_cols = business_df.columns[13:648]
print category_cols

attribute_cols = business_df.columns[648:737]
print attribute_cols

checkin_cols = business_df.columns[737:]
print checkin_cols

attribute_checkin_cols = business_df.columns[648:]
#business_df.info(verbose=True, null_counts=True)

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
get_ipython().run_line_magic('matplotlib', 'inline')

# decide map range based on min and max latitude/longitudes
margin = .01
lat_min = min(business_df['latitude'].values) - margin
lat_max = max(business_df['latitude'].values) + margin
lon_min = min(business_df['longitude'].values) - margin
lon_max = max(business_df['longitude'].values) + margin

# create map
m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min)/2,
            lon_0=(lon_max - lon_min)/2,
            projection='merc',
            resolution='h',
            area_thresh=10000.)
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='white', lake_color='#46bcec')
lons, lats = m(business_df['longitude'].values, business_df['latitude'].values)

# plot points colored by neighborhood
#col = business_df['neighborhood'].replace({u'': 'b'})
color_labels = business_df['neighborhood'].unique()
rgb_values = sns.color_palette("Set2", len(color_labels))
color_map = dict(zip(color_labels, rgb_values))
m.scatter(lons, lats, marker='o', c=business_df['neighborhood'].map(color_map), zorder=5)
plt.show()

sns.distplot(business_df['stars'], kde=False);

sns.boxplot(x='WiFi_free', y='stars', data=business_df);

min_ratings = 5
cats = business_df.columns[13:648].values  # category columns
upper_perc = np.percentile(business_df['stars'], 85)
lower_perc = np.percentile(business_df['stars'], 25)
cat_names = []
cat_means = []
for cat in cats:
    if business_df[business_df[cat] == 1].stars.value_counts().sum() >= min_ratings:
        curr_mean = business_df[business_df[cat] == 1].stars.mean()
        if curr_mean >= upper_perc or curr_mean <= lower_perc:
            cat_names.append(cat)
            cat_means.append(business_df[business_df[cat] == 1].stars.mean())

# plot without sorting
#ax = sns.barplot(x=cat_means, y=cats[:how_many], color='g')

# sort by least to most stars
yx = zip(cat_means, cat_names)
yx.sort()
y_sorted, x_sorted = zip(*yx)

fig, ax = plt.subplots()
fig.set_size_inches(8, len(cat_means)/4)
ax = sns.barplot(x=y_sorted, y=x_sorted, color='g')

boxplot_df = pd.DataFrame([])
for i, cat in enumerate(cats):
    currgroup = business_df[business_df[cat] == 1]
    if currgroup.stars.value_counts().sum() >= min_ratings:
        if currgroup.stars.mean() >= upper_perc or currgroup.stars.mean() <= lower_perc:
            stars_df = pd.DataFrame([])
            stars_df['Stars'] = currgroup.stars
            stars_df['Category'] = currgroup[cat].name
            stars_df['Mean'] = currgroup.stars.mean()
            boxplot_df = pd.concat([boxplot_df, stars_df])

boxplot_df = boxplot_df.sort_values(['Mean']).reset_index(drop=True)

fig, ax = plt.subplots()
fig.set_size_inches(8, boxplot_df['Category'].nunique()/4)
ax = sns.boxplot(x='Stars', y='Category', data=boxplot_df)

corr_df = business_df[['stars', 'Alcohol_full_bar', 'Smoking_yes', 'GoodForKids', 'Ambience_romantic', 'BikeParking']]
sns.heatmap(corr_df.corr());

# find the most represented categories
business_df.iloc[:, 13:648].sum().sort_values(ascending=False).head(10)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# just look at one business category
select_df = business_df[business_df['Restaurants'] == 1]
# just look at attribute columns
model_df = select_df[attribute_cols]  # use just attributes not basic data or check-in columns
model_df['is_open'] = business_df['is_open']  # add one basic data column; not sure why this causes error

feature_cols = model_df.columns
X = model_df
y = select_df.stars

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=1)
print X_train.shape
print X_test.shape

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
print 'Train RMSE:'
print np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))

y_pred_test = lr.predict(X_test)
print 'Test RMSE:'
print np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
print ''
print lr.intercept_
res = pd.DataFrame({'feature': feature_cols, 'coef': lr.coef_})
print res.sort_values(by=['coef'], ascending=False)

from sklearn.model_selection import KFold, cross_val_score

# just look at one business category
select_df = business_df[business_df['Restaurants'] == 1]
# just look at attribute columns
model_df = select_df[attribute_cols]  # use just attributes not basic data or check-in columns
model_df['is_open'] = business_df['is_open']  # add one basic data column; not sure why this causes error

feature_cols = model_df.columns
X = model_df
y = select_df.stars

lr = LinearRegression()
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
cross_val_scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=kfold)
print '10-fold RMSEs:'
print [np.sqrt(-x) for x in cross_val_scores]
print 'CV RMSE:'
print np.sqrt(-np.mean(cross_val_scores))  # RMSE is the sqrt of the avg of MSEs
print 'Std of CV RMSE:'
print np.std(cross_val_scores)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict

pf = PolynomialFeatures(degree=2,interaction_only=True)
X_pf = pf.fit_transform(X)  # only apply to attribute columns
print X.shape
print X_pf.shape

lr2 = LinearRegression()
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
pf_cross_val_scores = cross_val_score(lr2, X_pf, y, scoring='neg_mean_squared_error', cv=kfold)

pf_cross_val_predicts = cross_val_predict(lr2, X_pf, y, cv=kfold)
print pf_cross_val_predicts[:20]

print '10-fold RMSEs:'
print [np.sqrt(-x) for x in pf_cross_val_scores]
print 'CV RMSE:'
print np.sqrt(-np.mean(pf_cross_val_scores))  # RMSE is the sqrt of the avg of MSEs
print 'Std of CV RMSE:'
print np.std(pf_cross_val_scores)

from sklearn.linear_model import RidgeCV

ridge = RidgeCV(store_cv_values=True)
ridge.fit(X, y)

ridge_MSEs = ridge.cv_values_
print 'Ridge CV RMSE:'
print np.sqrt(np.mean(ridge_MSEs))

# try on expanded polynomial features
ridge_pf = RidgeCV(store_cv_values=True)
ridge_pf.fit(X_pf, y)

ridge_pf_MSEs = ridge_pf.cv_values_
print 'Ridge PF CV RMSE:'
print np.sqrt(np.mean(ridge_pf_MSEs))

from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz

# use business attributes to predict star rating, same as above
rg_X_train, rg_X_test, rg_y_train, rg_y_test = train_test_split(X, y, test_size=.2, random_state=1)
features = [x.encode('utf-8') for x in model_df.columns.values]

rg = DecisionTreeRegressor(max_depth=3)
rg.fit(rg_X_train, rg_y_train)
print 'RMSE:'
print np.sqrt(metrics.mean_squared_error(rg_y_test, rg.predict(rg_X_test)))
print ''

# visualize the tree
dot_data = StringIO()  
export_graphviz(rg, out_file=dot_data,  
                    feature_names=features,  
                    filled=True, rounded=True,  
                    special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

# most important features of the decision tree regressor
pd.DataFrame({'feature': model_df.columns.values, 'importance': rg.feature_importances_}).sort_values(by='importance', ascending=False).head()

depths = range(1,11)
train_rmse, test_rmse = [],[]
for depth in depths:
    decision_tree = DecisionTreeRegressor(max_depth=depth, random_state=1)
    decision_tree.fit(rg_X_train, rg_y_train)
    curr_train_rmse = np.sqrt(metrics.mean_squared_error(rg_y_train, decision_tree.predict(rg_X_train)))
    curr_test_rmse = np.sqrt(metrics.mean_squared_error(rg_y_test, decision_tree.predict(rg_X_test)))
    train_rmse.append(curr_train_rmse)
    test_rmse.append(curr_test_rmse)
sns.mpl.pyplot.plot(depths,train_rmse,label='train_rmse')
sns.mpl.pyplot.plot(depths,test_rmse,label='test_rmse')
sns.mpl.pyplot.xlabel("maximum tree depth")
sns.mpl.pyplot.ylabel("rmse - lower is better")
sns.mpl.pyplot.legend();

from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz

# use business attributes to predict star rating, same as above
feature_cols = [x.encode('utf-8') for x in model_df.columns.values]
X = model_df
y = select_df.stars

rg = DecisionTreeRegressor(max_depth=3, max_features=10)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
cross_val_scores = cross_val_score(rg, X, y, scoring='neg_mean_squared_error', cv=kfold)
print '10-fold RMSEs:'
print [np.sqrt(-x) for x in cross_val_scores]
print 'CV RMSE:'
print np.sqrt(-np.mean(cross_val_scores))  # RMSE is the sqrt of the avg of MSEs
print 'Std of CV RMSE:'
print np.std(cross_val_scores)

# visualize the tree
rg.fit(X, y)
dot_data = StringIO()
export_graphviz(rg, out_file=dot_data,  
                    feature_names=feature_cols,  
                    filled=True, rounded=True,  
                    special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

from sklearn.model_selection import GridSearchCV

# grid search to find best parameters
rg_grid = DecisionTreeRegressor(random_state=1)
max_depth_range = range(1, 11)
max_features_range = [x/20.0 for x in range(1, 20)]
param_grid = dict(max_depth=max_depth_range, max_features=max_features_range)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
grid = GridSearchCV(rg_grid, param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X, y)
#print grid.cv_results_['mean_test_score']
tree_model = grid.best_estimator_
print 'Best RMSE and parameters:'
print np.sqrt(-grid.best_score_), grid.best_params_
#for mean, param in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
#    print mean, param

scores = np.sqrt([-x for x in grid.cv_results_['mean_test_score']])  # convert to RMSE
max_f = [x.values()[0] for x in grid.cv_results_['params']]
max_d = [x.values()[1] for x in grid.cv_results_['params']]

rg_grid_results = pd.DataFrame({'RMSE': scores,
                                'max_features': max_f,
                                'max_depth': max_d})

# plot the results
sns.swarmplot(x='max_features', y='RMSE', data=rg_grid_results)
plt.show()
f2 = sns.swarmplot(x='max_depth', y='RMSE', data=rg_grid_results)
plt.show()

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR

# GridSearch to see if optimizing the parameters will improve (lower) the RMSE
reg_models = [('LinReg', LinearRegression(), {'normalize': [True, False]}),
              ('DecTreeReg', DecisionTreeRegressor(), {'max_depth': range(2, 10, 2), 'max_features': [0.25, 0.5, 0.75, 1.0]}),
              ('Lasso', Lasso(), {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}),
              ('Ridge', Ridge(), {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}),
              ('ElasticNet', ElasticNet(), {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}),
              ('SVR', SVR(), {'C': [1e0, 1e1, 1e2, 1e3], 'gamma': np.logspace(-2, 2, 5)})]

names = []
params = []
results = []
for name, model, param in reg_models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    reg_grid = GridSearchCV(model, param, cv=kfold, scoring='neg_mean_squared_error')
    reg_grid.fit(X, y)
    # just keep the results using the best parameters
    best_model = reg_grid.best_estimator_
    names.append(name)
    params.append(reg_grid.best_params_)
    results.append(np.sqrt(-reg_grid.best_score_))  # convert to RMSE

result_df = pd.DataFrame({'models': names, 'results': results})
result_df.columns = ['models', 'RMSE']
result_df.sort_values(by='RMSE', ascending=False, inplace=True)
print result_df.tail(1)

# plot results
sns.barplot(x='models', y='RMSE', data=result_df);

