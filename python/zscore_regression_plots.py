get_ipython().magic('matplotlib inline')

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from pymongo import MongoClient
import pymongo

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 10, 8

client = MongoClient("localhost", 27017)
db = client["microscopium"]

# get z scores and sample type (empty, control type, etc)
docs = db.samples.find({"screen":"KEEFE"}).sort([("_id", pymongo.ASCENDING)])

scores = {}

scores["average_EduAvgInten_normalised_to_OTP_robust_z_scored"] = []
scores["average_EduTotalInten_normalised_to_OTP_robust_z_scored"] = []
scores["average_NucSize_normalised_to_OTP_robust_z_scored"] = []
scores["average_NucArea_normalised_to_OTP_robust_z_scored"] = []
scores["average_NucSize_normalised_to_OTP_robust_z_scored"] = []
scores["average_cell_count_normalised_to_OTP_robust_z_scored"] = []

coding = []

for doc in docs:
    for key in scores.keys():
        scores[key].append(doc["overlays"][key])
        
    if doc["empty"] == True:
        coding.append('empty')
    elif doc["control_pos"] == True:
        coding.append("control_pos")
    elif doc["control_neg"] == True:
        coding.append("control_neg")
    else:
        coding.append("treated")

data = pd.read_csv("../microscopium-data/KEEFE-features.csv", index_col=0)
feature_cols = data.columns
data["type"] = coding

for score in scores.keys():
    data[score] = scores[score]

data_experimental = data[data["type"].isin(["treated"])]
scaler = StandardScaler()
data_experimental_std = scaler.fit_transform(data_experimental[feature_cols])

# fit simple linear regression models for each
def fit_models(data, scores):
    for key in scores.keys():
        X = data
        y = data_experimental[key]
        regression = LinearRegression()
        regression.fit(X, y)
        score = regression.score(X, y)
        print("{0}, {1}".format(key, score))
        
fit_models(data_experimental_std, scores)

data_pca_50 = PCA(n_components=50).fit_transform(data_experimental_std)
data_pca = PCA(n_components=2).fit_transform(data_experimental_std)
data_tsne = TSNE(n_components=2).fit_transform(data_pca_50)

fit_models(data_pca_50, scores)

fit_models(data_pca, scores)

fit_models(data_tsne, scores)

key = "average_EduAvgInten_normalised_to_OTP_robust_z_scored"
plt.scatter(data_pca[:, 0], data_pca[:, 1], s=45, c=data_experimental[key], cmap="viridis", vmin=-2, vmax=2)
plt.colorbar()

key = "average_EduAvgInten_normalised_to_OTP_robust_z_scored"
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], s=45, c=data_experimental[key], cmap="viridis", vmin=-2, vmax=2)
plt.colorbar()



