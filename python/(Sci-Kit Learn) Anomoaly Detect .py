import pandas as pd

rentals_df = pd.read_csv('rentals.csv', encoding='latin-1')
rentals_df.head()

stations_df = pd.read_csv('stations.csv', encoding='latin-1')
stations_df.head()

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# return Series of distance between each point and his distance with the closest centroid
def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance

# train markov model to get transition matrix
def getTransitionMatrix (df):
	df = np.array(df)
	model = msm.estimate_markov_model(df, 1)
	return model.transition_matrix

def markovAnomaly(df, windows_size, threshold):
    transition_matrix = getTransitionMatrix(df)
    real_threshold = threshold**windows_size
    df_anomaly = []
    for j in range(0, len(df)):
        if (j < windows_size):
            df_anomaly.append(0)
        else:
            sequence = df[j-windows_size:j]
            sequence = sequence.reset_index(drop=True)
            df_anomaly.append(anomalyElement(sequence, real_threshold, transition_matrix))
    return df_anomaly

# the hours and if it's night or day (7:00-22:00)
rentals_df['hours'] = rentals_df['Starttime_dt'].dt.hour
rentals_df['daylight'] = ((rentals_df['hours'] >= 7) & (rentals_df['hours'] <= 22)).astype(int)

# the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
rentals_df['DayOfTheWeek'] = rentals_df['Starttime_dt'].dt.dayofweek
rentals_df['WeekDay'] = (rentals_df['DayOfTheWeek'] < 5).astype(int)
# An estimation of anomly population of the dataset (necessary for several algorithm)
outliers_fraction = 0.01

# creation of 4 distinct categories that seem useful (week end/day week & night/day)
rentals_df['Starttime_cat'] = rentals_df['WeekDay']*2 + rentals_df['daylight']

a = rentals_df.loc[rentals_df['Starttime_cat'] == 0, 'Tripduration_mins']
b = rentals_df.loc[rentals_df['Starttime_cat'] == 1, 'Tripduration_mins']
c = rentals_df.loc[rentals_df['Starttime_cat'] == 2, 'Tripduration_mins']
d = rentals_df.loc[rentals_df['Starttime_cat'] == 3, 'Tripduration_mins']

fig, ax = plt.subplots(figsize=(15,8))
a_heights, a_bins = np.histogram(a)
b_heights, b_bins = np.histogram(b, bins=a_bins)
c_heights, c_bins = np.histogram(c, bins=a_bins)
d_heights, d_bins = np.histogram(d, bins=a_bins)

width = (a_bins[1] - a_bins[0])/6

ax.bar(a_bins[:-1], a_heights*100/a.count(), width=width, facecolor='blue', label='WeekEnd Night')
ax.bar(b_bins[:-1]+width, (b_heights*100/b.count()), width=width, facecolor='green', label ='WeekEndDayLight')
ax.bar(c_bins[:-1]+width*2, (c_heights*100/c.count()), width=width, facecolor='red', label ='WeekDay Night')
ax.bar(d_bins[:-1]+width*3, (d_heights*100/d.count()), width=width, facecolor='black', label ='WeekDay DayLight')

ax.set_xlabel('Trip Duration in minutes')
plt.legend()
plt.show()

### Now we have to be careful in choice of outlier detector. What is the nature of our data? It is partly 
### unordered (Bike IDs, Trip IDs, Stations) and partly ordered (timestamps of start and stop of trips)

### Let us start with simple clustering

rentals_df['Starttime_num']=mdates.date2num(rentals_df['Starttime_dt'].astype(datetime))
rentals_df['Stoptime_num']=mdates.date2num(rentals_df['Stoptime_dt'].astype(datetime))

# pull out data for PCA analysis
data = rentals_df[['Tripduration', 'Starttime_num', 'Stoptime_num', 'From station id', 'To station id',                    'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# reduce to 2 importants features
pca = PCA(n_components=2)
data = pca.fit_transform(data)
# standardize these 2 new features
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)

# calculate with different number of centroids to see the loss plot (elbow method)
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]
fig, ax = plt.subplots()
ax.plot(n_cluster, scores)
plt.show()

# choose 15 centroids arbitrarily and add these data to the central dataframe
rentals_df['cluster'] = kmeans[14].predict(data)
rentals_df['principal_feature1'] = data[0]
rentals_df['principal_feature2'] = data[1]
rentals_df['cluster'].value_counts()

#plot the different clusters with the 2 main features
fig, ax = plt.subplots()
colors = {0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown',           9:'purple', 10:'white', 11: 'grey', 12:'lightblue', 13:'lightgreen', 14: 'darkgrey'}
ax.scatter(rentals_df['principal_feature1'], rentals_df['principal_feature2'], 
           c=rentals_df["cluster"].apply(lambda x: colors[x]))
plt.show()

# lets zoom in separately to see clearer
fig, ax = plt.subplots(1,2,figsize=(15,8))
colors = {0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown',           9:'purple', 10:'white', 11: 'grey', 12:'lightblue', 13:'lightgreen', 14: 'darkgrey'}
ax[0].scatter(rentals_df['principal_feature1'], rentals_df['principal_feature2'], 
           c=rentals_df["cluster"].apply(lambda x: colors[x]))
ax[0].set_xlim(-2,4.1)
ax[0].set_ylim(-2,5)
ax[0].set_title('PCA Analysis (Lower quadrant)')
ax[1].scatter(rentals_df['principal_feature1'], rentals_df['principal_feature2'], 
           c=rentals_df["cluster"].apply(lambda x: colors[x]))
ax[1].set_xlim(40,55)
ax[1].set_ylim(50,65)
ax[1].set_title('PCA Analysis (Upper quadrant)')
plt.show()

# get the distance between each point and its nearest centroid. 
# The biggest distances are considered as anomalies
distance = getDistanceByPoint(data, kmeans[14])
number_of_outliers = int(outliers_fraction*len(distance))
threshold = distance.nlargest(number_of_outliers).min()
rentals_df['anomaly21'] = (distance >= threshold).astype(int)

# visualisation of anomaly with cluster view
fig, ax = plt.subplots()
colors = {0:'blue', 1:'red'}
ax.scatter(rentals_df['principal_feature1'], rentals_df['principal_feature2'],            c=rentals_df["anomaly21"].apply(lambda x: colors[x]))
ax.set_title('Outlier PCA Analysis')
plt.show()

anomalous_rides=rentals_df[rentals_df['anomaly21']==1]

# repeat visualization with zooms, as before
fig, ax = plt.subplots(1,2,figsize=(15,8))
colors = {0:'blue', 1:'red'}
ax[0].scatter(rentals_df['principal_feature1'], rentals_df['principal_feature2'], 
           c=rentals_df["anomaly21"].apply(lambda x: colors[x]))
ax[0].set_xlim(-2,4.1)
ax[0].set_ylim(-2,5)
ax[0].set_title('Outlier PCA Analysis (Lower quadrant)')
ax[1].scatter(rentals_df['principal_feature1'], rentals_df['principal_feature2'], 
           c=rentals_df["anomaly21"].apply(lambda x: colors[x]))
ax[1].set_xlim(40,55)
ax[1].set_ylim(50,65)
ax[1].annotate(anomalous_rides['Trip id'],xy=(anomalous_rides['principal_feature1'],anomalous_rides['principal_feature2']))
ax[1].set_title('Outlier PCA Analysis (Upper quadrant)')

plt.show()

anomalous_rides.head()

# Now lets try an isolation-forest algorithm, from Scikit-Learn
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# pull out data for Isolation Forests analysis
data = rentals_df[['Tripduration', 'Starttime_num', 'Stoptime_num', 'From station id', 'To station id',                    'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)

# fit the model
clf = IsolationForest(contamination = outliers_fraction)
clf.fit(data)
rentals_df['anomaly_if'] = pd.Series(clf.predict(data))
rentals_df['anomaly_if'] = rentals_df['anomaly_if'].map( {1: 0, -1: 1} )
print(rentals_df['anomaly_if'].value_counts())

bikeid_min=rentals_df['Bikeid'].min()
bikeid_max=rentals_df['Bikeid'].max()
tripid_min=rentals_df['Trip id'].min()
tripid_max=rentals_df['Trip id'].max()
fromstationid_min=rentals_df['From station id'].min()
fromstationid_max=rentals_df['From station id'].max()
tostationid_min=rentals_df['To station id'].min()
tostationid_max=rentals_df['To station id'].max()

print('Bike ID={}:{} | Trip ID={}:{} | From ID={}:{} | To ID={}:{}'.format(bikeid_min,bikeid_max,                                    tripid_min,tripid_max,fromstationid_min, fromstationid_max,                                    tostationid_min, tostationid_max))

# plot the line, the samples, and the nearest vectors to the plane
#xx, yy = np.meshgrid(np.linspace(fromstationid_min, fromstationid_max, 50), \
#                     np.linspace(tostationid_min, tostationid_max, 50))
rand_smpl_xx = [ rentals_df['From station id'][i] for i in                 sorted(np.random.choice(rentals_df['From station id'], size=50, replace=False)) ]
rand_smpl_yy = [ rentals_df['To station id'][i] for i in                 sorted(np.random.choice(rentals_df['To station id'], size=50, replace=False)) ]
xx, yy = np.meshgrid(rand_smpl_xx, rand_smpl_yy)
xxyy_combo = np.c_[xx.ravel(), yy.ravel()]
Z = clf.decision_function(xxyy_combo)
Z = Z.reshape(xx.shape)

plt.title("Isolation Forest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

a1 = rentals_df.loc[rentals_df['anomaly_if'] == 0, ['From station id', 'To station id']] # normal
a2 = rentals_df.loc[rentals_df['anomaly_if'] == 1, ['From station id', 'To station id']] #anomaly

b1 = plt.scatter(a1['From station id'], a1['To station id'], c='green', s=20, edgecolor='k')
b2 = plt.scatter(a2['From station id'], a2['To station id'], c='red', s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((fromstationid_min, fromstationid_max))
plt.ylim((tostationid_min, tostationid_max))
plt.legend([b1, b2],
           ["regular data",
            "anomalies"],
           loc="upper left")
plt.show()

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
xxyy_combo=np.c_[xx.ravel(), yy.ravel()]
print(xxyy_combo)
Z = clf.decision_function(xxyy_combo)
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()

((52*52)-52)/2

station_df = pd.read_csv('stations.csv', encoding='latin-1')

station_df.head()

rentals_df.groupby(['From station id','To station id']).size().reset_index().rename(columns={0:'count'})

rentals_df.drop_duplicates(subset=['From station id','To station id'])

