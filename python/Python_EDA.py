import pandas as pd
import numpy as np

airbnb = pd.read_csv("airbnb_manhattan.csv")
airbnb2 = pd.read_csv("airbnb_manhattan2.csv")
airbnb = pd.concat([airbnb,airbnb2])
airbnb.drop_duplicates(subset='roomID', keep='first', inplace=True)

airbnb.shape

from matplotlib import pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

airbnb.columns.tolist()

airbnb.head

type(airbnb)

x =plt.hist(airbnb['price'],bins= 200)
plt.xlabel('Cost per night')
plt.ylabel('Number of Listings')
plt.title('Cost per Night', fontsize=20)
plt.xlim(0,1200)    # set the ylim to ymin, ymax

import plotly
# connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected=True)

from plotly.graph_objs import Figure, Histogram, Layout

min_ = airbnb['price'].min()
max_ = airbnb['price'].max()

data = [Histogram(x=airbnb['price'], 
                  xbins=dict(start=min_,
                             end=max_,
                             size=(max_-min_)/100))]
layout = Layout(title="Costs",
                bargap=0.2)
fig = Figure(data=data, layout=layout)

plotly.offline.iplot(fig, show_link=False, image_width=600, image_height=400)

from plotly.graph_objs import Scatter


data = [Scatter(x=airbnb['numRooms'], y=airbnb['price'], mode = 'markers')]#, text=df['movie_title'])]
layout = Layout(title="Price versus number of rooms")

fig = Figure(data=data, layout=layout)

plotly.offline.iplot(fig, show_link=False)



airbnb.isnull().sum(axis=0)

airbnbNew = airbnb[pd.notnull(airbnb['checkin'])]
airbnbNew.isnull().sum(axis=0)

airbnbNew.columns.to_series().groupby(airbnbNew.dtypes).groups

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(airbnbNew['numGuests'], airbnbNew['numRooms'], airbnbNew['price'])
plt.show()

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
plot_corr(airbnbNew)

print(airbnbNew.iloc[:,8:18])

airbnbNew.drop(['responseTimeShown', 'roomType','roomID','bedType'], axis=1)

#['bathType', 'bedType', 'bedroomType', 'shortDesc'], dtype='object')}

airbnbNew.bathType.unique() # need to take out s in two of them, convert to string
airbnbNew.bedroomType.unique() #bedroom(s), need to take out s, convert to string
airbnbNew.shortDesc.unique() #fine

airbnbNew.loc[:,'bathType'] = [word[:-1] if word[-1]=="s" else word for word in airbnbNew['bathType']]
airbnbNew.loc[:,'bedroomType'] = [word[:-1] if word[-1]=="s" else word for word in airbnbNew['bedroomType']]

bathDF = pd.get_dummies(airbnbNew['bathType'])
bedroomDF = pd.get_dummies(airbnbNew['bedroomType'])
shortDescDF = pd.get_dummies(airbnbNew['shortDesc'])

airbnbNew = pd.concat([airbnbNew, bathDF, bedroomDF, shortDescDF],axis=1, join_axes=[airbnbNew.index])

# Dropping all the non numeric columns...or the ones with a ton of NAs (cough...host reviews...cough)
airbnbNew.drop(['bathType','roomType','bedroomType','shortDesc','bedType','responseTimeShown','numHostReviews'], axis=1, inplace=True)
airbnbNew.set_index('roomID')

airbnbNew['isSuperhost'] = (airbnbNew['isSuperhost'] == True).astype(int)
airbnbNew = airbnbNew[airbnbNew['price'] < 700]
print(airbnbNew['price'].max)
airbnbNew

print(airbnbNew['price'].max())

#Normalizing:
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(airbnbNew)
airbnb_Norm = pd.DataFrame(np_scaled)

# airbnb_Norm['roomID'] = Series(airbnbNew['roomID'], index=airbnb_Norm.index)
# airbnb_Norm.set_index('roomID')
# airbnbNew.isnull().sum()
# Entire home/apt  Private room  Shared room 

# listNames = list(airbnbNew.columns.values)
airbnb_Norm.columns = listNames


print(airbnb_Norm.head(5))
print(airbnb_Norm.columns.values)

plot_corr(airbnb_Norm)

import seaborn as sns

cols = airbnb_Norm.columns.tolist()
#print(cols)
cols = [cols[14]] + cols[:-14]+cols[15:]
print(cols)
airbnb_Norm = airbnb_Norm[cols]

corr = airbnb_Norm.corr()
f, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, cmap="YlGnBu",
           square=True, ax=ax)
plt.show()

airbnbNew = airbnbNew.set_index('roomID')

from sklearn import svm, datasets, cross_validation
from sklearn import metrics 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Shuffling the dataframe
airbnbNew = airbnbNew.iloc[np.random.permutation(len(airbnbNew))]

# Finally dropping off roomID with roomID as new index
# airbnbNew.drop(['roomID'], axis=1, inplace=True)

# Removing the price column from the matrix.
airbnb_NormNoY = airbnb_Norm.drop(['price'], axis=1, inplace=False)
airbnb_Matrix = airbnb_NormNoY.as_matrix()
X = airbnb_Matrix
y = airbnbNew['price']

# Split the data into training/testing sets
airbnb_X_train = airbnb_Matrix[:-200]
airbnb_X_test = airbnb_Matrix[-200:]

# Split the targets into training/testing sets
airbnb_y_train = y[:-200]
airbnb_y_test = y[-200:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(airbnb_X_train, airbnb_y_train)

# Make predictions using the testing set
airbnb_y_pred = regr.predict(airbnb_X_test)

print('Variance score: %.2f' % r2_score(airbnb_y_test, airbnb_y_pred))
print("Mean squared error: %.2f"
      % mean_squared_error(airbnb_y_test, airbnb_y_pred))

zippedResult = list(zip(airbnb_y_test, airbnb_y_pred))

i = 0
len1 = len(airbnbNew)
for result in zippedResult:
    print(result)
    print(airbnbNew.index[len1-1-i])
    i += 1
print(len(zippedResult))

import numpy
print(numpy.__version__)
import sys
print(sys.path)

import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#from sklearn.multioutput import MultiOutputRegressor

max_depth = 20
#regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
#                                                          random_state=0))
#regr_multirf.fit(airbnb_X_train, airbnb_y_train)

regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=50)
regr_rf.fit(airbnb_X_train, airbnb_y_train)

# Predict on new data
#y_multirf = regr_multirf.predict(airbnb_X_test)
airbnb_y_pred_rf = regr_rf.predict(airbnb_X_test)

zippedResult = list(zip(airbnb_y_test, airbnb_y_pred_rf))

i = 0
len1 = len(airbnbNew)
for result in zippedResult:
    print(result)
    print(airbnbNew.index[len1-1-i])
    i += 1

print('Variance score: %.2f' % r2_score(airbnb_y_test, airbnb_y_pred_rf))
print("Mean squared error: %.2f" % mean_squared_error(airbnb_y_test, airbnb_y_pred_rf))

import matplotlib  
import matplotlib.pyplot as plt  
import pandas as pd
#Inline Plotting for Ipython Notebook 
get_ipython().run_line_magic('matplotlib', 'inline')

#pd.options.display.mpl_style = 'default' #Better Styling  
new_style = {'grid': False} #Remove grid  
matplotlib.rc('axes', **new_style)  
from matplotlib import rcParams  
rcParams['figure.figsize'] = (17.5, 17) #Size of figure  
rcParams['figure.dpi'] = 250

print(airbnb['longitude'].head(5))

P=airbnbNew.plot(kind='scatter', x='longitude', y='latitude',color='white',
                 xlim=(-74.06,-73.9),ylim=(40.67, 40.85),s=5,alpha=1)
P.set_facecolor('black') #Background Color

airbnbNew.to_csv("airbnbNew_Data.csv")
airbnb.to_csv("airbnb_Data.csv")
airbnb_NormNoY.to_csv("airbnb_NormNoY_Data.csv")
airbnb_Norm.to_csv("airbnb_Norm_Data.csv")



