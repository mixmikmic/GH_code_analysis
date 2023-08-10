import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')

raw_data = pd.read_csv('epi_r.csv')

list(raw_data.columns)

raw_data.rating.describe()

# sns.set_style('darkgrid')
# raw_data.rating.hist(bins=20)
# plt.title('Histogram of Recipe Ratings');

# Count nulls 
null_count = raw_data.isnull().sum()
null_count[null_count>0]

# Took too long to keep restarting so I just commented out the long processing code. 
from sklearn.svm import SVR
svr = SVR()
X = raw_data.drop(['rating', 'title', 'calories', 'protein', 'fat', 'sodium'], 1)
Y = raw_data.rating
# svr.fit(X,Y)

#  plt.scatter(Y, svr.predict(X));

# svr.score(X,Y)

from sklearn.model_selection import cross_val_score
# cross_val_score(svr, X, Y, cv=5)

# Clean the data and remove the missing data
raw_data = raw_data.dropna()

# Checked that we didn't lose too much data after dropna.
# Count is 15,862 compared to 20,052 without dropping data.
print(raw_data.rating.describe())

# Check the breakdown of the ratings.
print('\n')
print(raw_data['rating'].value_counts())

# Create a new categorical column called 'Good_Rating'
raw_data['Good_Rating'] = np.where(raw_data['rating']>4.0,1,0)

# Check to see data type.
raw_data.dtypes

# Split the data.
X = raw_data.drop(['title', 'rating', 'Good_Rating'], 1)
Y = raw_data['Good_Rating']

from sklearn.feature_selection import SelectKBest, f_classif
# Use SelectKBest to obtain the top 30 features.
select_k = SelectKBest(f_classif, k=30)

# Fit the data
fit = select_k.fit(X, Y)

# Get the new x data points from the selection
selected_features = fit.get_support(indices=True)

# Match up points to the names
k_features = X[X.columns[selected_features]]

from sklearn.svm import SVC

# Build the SVC model.
svc = SVC()

# Fit the data.
svc.fit(k_features, Y)

from sklearn.model_selection import cross_val_score

# Test accuracy of data.
cross_val_score(svc, k_features, Y, cv=5)



