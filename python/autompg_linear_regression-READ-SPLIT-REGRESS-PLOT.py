get_ipython().magic('matplotlib inline')
from sml import execute

query = 'READ "../data/auto-mpg.csv" (separator = "\s+", header = None) AND  REPLACE (missing = "?", strategy = "mode") AND SPLIT (train = .8, test = .2, validation = .0) AND   REGRESS (predictors = [2,3,4,5,6,7,8], label = 1, algorithm = simple) AND   PLOT (modelType="AUTO", plotTypes="AUTO")'

execute(query, verbose=True)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve, validation_curve

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize']=(12,12)
sns.set()

#Names of all of the columns
names = [
       'mpg'
    ,  'cylinders'
    ,  'displacement'
    ,  'horsepower'
    ,  'weight'
    ,  'acceleration'
    ,  'model_year'
    ,  'origin'
    ,  'car_name'
]

#Import dataset
data = pd.read_csv('../data/auto-mpg.csv', sep = '\s+', header = None, names = names)

data.head()

# Remove NaNs
data_clean=data.applymap(lambda x: np.nan if x == '?' else x).dropna()

# Sep Predictiors From Labels
X = data_clean[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', "origin"]]

#Select target column
y = data_clean['mpg']

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# Define and train  linear regression model
estimator = linear_model.LinearRegression()

# Generate Learning Cures
train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train) 

# Train Linear Regression Model
estimator.fit(X_train, y_train)

# Generate Validation Curves
param_range = np.arange(0, 5)

v_train_scores, v_test_scores = validation_curve(estimator, X_test, y_test, param_name='normalize', param_range=param_range)

score = estimator.score(X_test, y_test)
print('Accuracy :', score)

g = sns.PairGrid(data_clean, palette='PuOr_r')
#g.map(sns.kdeplot, cmap="PuOr_r") # If I try to do this, and then plot diag, orginal plot remains....

g = g.map_diag(sns.kdeplot, shade=True) # can't add color arg...
# To scale diag imgs, must go into indvidual scale...

g = g.map_upper(sns.kdeplot, cmap='PuOr_r')
g = g.map_lower(sns.kdeplot, cmap='PuOr_r')

plt.show()
plt.close()



# Color Blindness Avoid:
# Green & Red & Brown & Blue.
# Blue & Purple.
# Light Green & Yellow.
# Blue & Grey.
# Green & Grey.
# Green & Black.
# Columns that the user wants...
columns = [0,1,2,3]


color_pal = ['purple', 'dark green', 'orange', 'grey'] # For 1-D KDE
cmap_pal = ['PuOr_r'] # For 2-D KDE
classes = [] # May not have a class for categories

column_headers =  data_clean.columns.values.tolist() # Grab headers from df
column_headers = [column_headers[x] for x in columns] # Map headers to indices selected

fig, ax = plt.subplots(len(columns), len(columns))

if not classes:
    for col1, i in enumerate(columns):
        for col2, j in enumerate(columns):

            if i == j:
                sns.kdeplot(data_clean[data_clean.columns[col1]], ax=ax[col1][col2], color=color_pal[0], shade=True, legend=False)
            else:
                sns.kdeplot( data_clean[data_clean.columns[col1]], data_clean[data_clean.columns[col2]], ax=ax[col1][col2], cmap=cmap_pal[0])

            # Formatting
            if j == 0:
                ax[i,j].set_xticklabels([])
                ax[i,j].set_ylabel(column_headers[i])
                ax[i,j].set_xlabel('')
                if i == len(columns)-1:
                    ax[i,j].set_xlabel(column_headers[j])
            elif i == len(columns)-1:
                ax[i,j].tick_params(axis='y', which='major', bottom='off')
                ax[i,j].set_yticklabels([])
                ax[i,j].set_xlabel(column_headers[j])
                ax[i,j].set_ylabel('')                
            else:
                ax[i,j].set_xticklabels([])
                ax[i,j].set_xlabel('')
                
                ax[i,j].set_yticklabels([])
                ax[i,j].set_ylabel('')
                
plt.show()
plt.close()

plt.figure()
plt.xlabel("Training examples")
plt.ylabel("Score")

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="orange")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="purple")
plt.plot(train_sizes, train_scores_mean, 'o-', color="orange",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="purple",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()
plt.close()

plt.figure()
plt.xlabel("Validation examples")
plt.ylabel("Score")

v_train_scores_mean = np.mean(v_train_scores, axis=1)
v_train_scores_std = np.std(v_train_scores, axis=1)
v_test_scores_mean = np.mean(v_test_scores, axis=1)
v_test_scores_std = np.std(v_test_scores, axis=1)

plt.fill_between(param_range, v_train_scores_mean - v_train_scores_std,
                 v_train_scores_mean + v_train_scores_std, alpha=0.1,
                 color="orange")
plt.fill_between(param_range, v_test_scores_mean - v_test_scores_std,
                 v_test_scores_mean + v_test_scores_std, alpha=0.1, color="purple")

plt.plot(param_range, v_train_scores_mean, 'o-', color="orange",
         label="Training score")

plt.plot(param_range, v_test_scores_mean, 'o-', color="purple",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()
plt.close()



