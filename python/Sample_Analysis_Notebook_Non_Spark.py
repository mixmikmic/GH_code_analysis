# This makes it so that matplotlib graphics will show up within the Jupyter Notebook.
get_ipython().magic('matplotlib inline')

# Standard library import
import re

# Data Analysis Tools
import pandas as pd
import numpy as np

# Visualization Tools
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Tools
import sklearn
import sklearn.mixture as mix
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans


def get_sensor_names(tag_names):
    """ Get tagnames starting with sensor.
    
    :param tag_names: Input time series data frame
    
    :return list of string tag names starting with sensor.
    """
    
    return [tag_name for tag_name in tag_names if re.search('^sensor.',tag_name)]

#Tag Names provided in data set description
tag_names = ['unit','cycle_num', 'setting1','setting2',
         'setting3', 'sensor1', 'sensor2',
         'sensor3', 'sensor4', 'sensor5', 'sensor6',
         'sensor7', 'sensor8', 'sensor9', 'sensor10',
         'sensor11', 'sensor12', 'sensor13', 'sensor14',
         'sensor15', 'sensor16', 'sensor17', 'sensor18',
         'sensor19', 'sensor20', 'sensor21']

train_data = pd.read_csv('train.txt', sep='\s+', header=None, names=tag_names)
train_data.head()

train_data.shape

train_data.describe()

train_data[['setting1', 'setting2', 'setting3']].describe()

plt.rcParams['figure.figsize'] = (10, 10)

# The sublot command takes arguments(row_col_pos) ... the first example 
# below says prepare a grid of 5 rows, column, and then draw the 
# next plot in the first position

plt.subplot(511)
sns.distplot(train_data['setting1'])
plt.subplot(513)
sns.distplot(train_data['setting2'])
plt.subplot(515)
sns.distplot(train_data['setting3'])

scaled_settings = train_data[['setting1', 'setting2', 'setting3']].apply(scale, axis=0)

sns.pairplot(scaled_settings)

scores= []
cluster_sizes = range(2,10)

#Set a seed value for the random number enerator to get repeatable results
np.random.seed(10)

for num_clusters in cluster_sizes:
    labels = KMeans(n_clusters=num_clusters, 
                    n_init=20, 
                    max_iter=1000).fit_predict(scaled_settings)
    
    silh_score = silhouette_score(scaled_settings.values, labels, sample_size=2000)
    scores.append(silh_score)

chosen_cluster_size = cluster_sizes[np.argmax(scores)]
chosen_cluster_size

plt.plot(cluster_sizes, scores)

predictions = KMeans(n_clusters=chosen_cluster_size).fit_predict(scaled_settings)

train_data['overall_setting'] = predictions
train_data.drop(['setting1', 'setting2', 'setting3'], axis=1, inplace=True)
train_data['overall_setting']= train_data['overall_setting'].astype('category')

import re

def get_sensor_names(tag_names):
    return [tag_name for tag_name in tag_names if re.search('^sensor.',tag_name)]
    

sensor_names = get_sensor_names(tag_names)

def is_zero(series):
    return series == 0

variance_by_setting = train_data[sensor_names + ['overall_setting']].groupby('overall_setting').var()
sensor_variance = variance_by_setting.apply(is_zero, axis=0).all()
sensor_variance

# Generate a list of the sensors with zero variance.
sensor_drop_list=sensor_variance[sensor_variance].index.values

# Drop these sensors from the training_data
train_data.drop(sensor_drop_list, axis=1, inplace=True)

sensor_drop_list

sensor_names = get_sensor_names(train_data)

train_data[sensor_names].corr()

#Note below was determined visually
high_corr_cols = ['sensor3','sensor4','sensor6',
                  'sensor7','sensor12','sensor17',
                  'sensor20']
high_corr_cols

train_data[sensor_names].corr()

train_data.drop(high_corr_cols, axis=1, inplace=True) 

sensor_names = get_sensor_names(train_data)

train_data[sensor_names].corr()

from sklearn.decomposition import PCA

#scale the training data
train_data[sensor_names]=train_data[sensor_names].apply(scale)

pca = PCA(n_components=0.99)
pc_fit = pca.fit(train_data[sensor_names])
pc_fit.explained_variance_ratio_.cumsum()

scores = pc_fit.transform(train_data[sensor_names].apply(scale))
sc_df = pd.DataFrame(scores, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
sc_df['setting']= train_data['overall_setting'].values
sc_df['unit'] = train_data['unit'].values
sc_df['cycle_num']=train_data['cycle_num'].values
sc_df.describe()

sc_df['SS_PC'] = sc_df[['PC1', 'PC2', 'PC3', 'PC4']].apply(lambda x: x**2).apply(sum, axis=1).apply(np.sqrt)

sc_df

orig = pc_fit.inverse_transform(scores)

sc = pd.DataFrame(orig, columns = train_data[sensor_names].columns)
sc['setting']= train_data['overall_setting'].values
sc['unit'] = train_data['unit'].values
sc['cycle_num']=train_data['cycle_num'].values

# Reshape the data to make it easier to plot in a matrix format.
# This involves stacking the variables.
sc_plot_data = pd.melt(sc_df, id_vars=['setting', 'unit', 'cycle_num'], 
                       value_vars=['PC1', 'PC2', 'PC3','PC4'], value_name='pc_value')
g = sns.FacetGrid(sc_plot_data, row="setting", col="variable")
g = g.map(plt.scatter, "cycle_num", "pc_value", edgecolor="w")

sc.head()

unit=1
sensor_names = get_sensor_names(sc)

#Need to fix the y scale
sc_plot_data = pd.melt(sc, id_vars=['setting', 'unit', 'cycle_num'], 
                       value_vars=sensor_names, value_name='value')

g = sns.FacetGrid(sc_plot_data[sc_plot_data['unit']==unit], col="setting", row="variable")
g = g.map(plt.scatter, "cycle_num", "value", edgecolor="w")

sc = pd.DataFrame(scores, columns = ['PC1', 'PC2', 'PC3'])
sc['setting']= train_data['overall_setting'].values
sns.pairplot(sc,hue = "setting",markers=".")

preprocessed_data = train_data.copy()

sns.pairplot(preprocessed_data[['setting1', 'setting2', 'setting3']].apply(scale, axis=0))

import matplotlib.pyplot as plt

train_data['pc1'] = pc_fit[:,0]
train_data['pc2'] = pc_fit[:,1]
train_data['pc3'] = pc_fit[:,2]


plt.plot(train_data.cycle_num[0:1000,train_data.unit==1], train_data.pc3[0:1000,train_data.unit==1])

sns.pairplot(train_data[['pc1', 'pc2', 'pc3']])



