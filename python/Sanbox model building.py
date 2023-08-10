import pandas as pd
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import cPickle as pk
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans

get_ipython().magic('matplotlib inline')

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/4app_train.pkl') as f:
    four_app_train = pk.load(f)
with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/4app_test.pkl') as f:
    four_app_test = pk.load(f)

four_app_train.fillna(value = 0,inplace = True)
four_app_test.fillna(value = 0,inplace = True)
four_app_test.head()

X_train = four_app_train[['channel_12','channel_12_diff']].values
X_test = four_app_test[['channel_12','channel_12_diff']].values
X_train.shape

def fit_Kmeans(X_train,X_test, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters).fit(X_train)
    centroids = kmeans.cluster_centers_ 
    pred_train = kmeans.predict(X_train)
    pred_test = kmeans.predict(X_test)
    SSE_train = np.sum([(X_train[idx] - centroids[pred_train[idx]])**2 for idx in xrange(len(X_train))])
    SSE_test = np.sum([(X_test[idx] - centroids[pred_test[idx]])**2 for idx in xrange(len(X_test))])
    
    return kmeans, SSE_train, SSE_test


cluster_list = xrange(1,8)
train_error = []
test_error = []

for c in cluster_list:
    _, SSE_train, SSE_test = fit_Kmeans(X_train,X_test, c)
    train_error.append(SSE_train)
    test_error.append(SSE_test)

plt.plot(cluster_list, train_error)
plt.plot(cluster_list, test_error)
plt.show()

n_states = 2
print "fitting to HMM and decoding ..."

# Make an HMM instance and execute fit
model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=10000)
model.fit(X_train)
state_means = model.means_
print state_means
# Predict the optimal sequence of internal hidden state
hidden_states_train = model.predict(X_train)
hidden_states_test = model.predict(X_test)


print "done"

def HMM_accuracy(obs_levels,hidden_states,state_means):
    predict_levels = [state_means[state] for state in hidden_states]
    test_error = 1 - (np.sum(obs_levels) - np.sum(predict_levels))/np.sum(obs_levels)
    return test_error

state_means = state_means[:,0].flatten()
print HMM_accuracy(X_test[:,0],hidden_states_test,state_means)

from BuildModel import HMM

with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/4app_train.pkl') as f:
    four_app_train = pk.load(f)
with open('/Users/nelly/Galvanize/Capstone/Electricity-load-prediction/data/house_1/4app_test.pkl') as f:
    four_app_test = pk.load(f)

four_app_train.fillna(value = 0,inplace = True)
four_app_test.fillna(value = 0,inplace = True)
X_train = four_app_train[['channel_12']].values
X_test = four_app_test[['channel_12']].values
my_HMM = HMM(X_test, X_train,2)
my_HMM.run()


apps = four_app_train.columns[::2]

for app in apps:
    X_train = four_app_train[[app]].values
    X_test = four_app_test[[app]].values
    my_HMM = HMM(X_test, X_train, 2)
    print "Model fitting for {} \n".format(app)
    my_HMM.run()
    print "\n"

X_train = four_app_train[['channel_5']].values
X_test_12 = four_app_test[['channel_12']].values
X_test_5 = four_app_test[['channel_5']].values
X_test_6 = four_app_test[['channel_6']].values
X_test_7 = four_app_test[['channel_7']].values

my_HMM = HMM(X_test, X_train, 2)
my_HMM.fit_HMM()

print "channel_12 LogProb: ", my_HMM.model.score(X_test_12)
print "channel_5 LogProb: ", my_HMM.model.score(X_test_5)
print "channel_6 LogProb: ", my_HMM.model.score(X_test_6)
print "channel_7 LogProb: ", my_HMM.model.score(X_test_7)



