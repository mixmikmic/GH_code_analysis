import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# need parallelism for training
from multiprocessing import Pool
_pool = Pool(processes=5)

# configure plotting
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12

# start with raw luminance
snowOnFile  = "../images/ASO/0_20170401_175413_1024_cropped.pkl"
snowOffFile = "../images/ASO/0_20160620_191028_0c79_cropped.pkl"
snowOnData  = pd.read_pickle(snowOnFile)
snowOffData = pd.read_pickle(snowOffFile)

# add class labels
snowOnData['label'] = 1
snowOffData['label'] = 0

# merge and shuffle
data = pd.concat([snowOnData, snowOffData], axis=0)
data = data.sample(frac=1)
print("{:d} | 0: {:d}, 1: {:d}".format(len(data),
                                      len(data[data.label == 0]), 
                                      len(data[data.label == 1])))

bands = ['band1', 'band2', 'band3', 'band4']
sample = data.sample(5000)
X = np.array(sample[bands])
y = np.array(sample.label)
X = StandardScaler().fit_transform(X)

X.shape

# we'll define the different classifiers to  test here. 
simple_gpc = {
    'n_jobs' : -1, # use threads
}

kernel_gpc_noopt = {
    'n_jobs'    : -1, 
    'kernel'    : 1.0 * kernels.RBF(length_scale=1.0),
    'optimizer' : None
}

kernel_gpc_opt = {
    'n_jobs' : -1, 
    'kernel' : 1.0 * kernels.RBF(length_scale=1.0)
    # default optimizer
}

paramsets = [kernel_gpc_opt]#, kernel_gpc_noopt, kernel_gpc_opt]

accuracies = []
folder = StratifiedKFold(5)
for params in paramsets:
    gpc = GaussianProcessClassifier(**params)
    acc = cross_val_score(gpc, X, y, cv=folder, n_jobs=-1)
    accuracies.append(np.mean(acc))
        
print(accuracies)

model_params = kernel_gpc_opt
model = GaussianProcessClassifier(**model_params)
model.fit(X, y)

model_outfile_name = "snowcover-model-ASO.pkl"
#joblib.dump(model, model_outfile_name)
model = joblib.load(model_outfile_name)

get_ipython().system('du -hc snowcover-model-ASO.pkl')

model.kernel_


X_all = X
y_all = X
_, snowOnProbs = zip(*model.predict_proba(X_all))

# which bands to plot?
b1 = 1
b2 = 3
preds = snowOnProbs

cmap = plt.get_cmap("coolwarm_r")

binary_colors = np.array([cmap(0), cmap(1)])

fig, axes = plt.subplots(1, 2)
ax1, ax2 = axes
s1 = ax1.scatter(list(X[:,b1]), list(X[:,b2]), c = preds, cmap=cmap, alpha=0.8, vmin=0, vmax=1)
ax1.set_title("Predictions (Probability)")
ax1.set_xlabel("Band {}".format(b1+1))
ax1.set_ylabel("Band {}".format(b2+1))
fig.colorbar(s1, orientation='horizontal', ax=axes.ravel().tolist())

ax2.scatter(list(X[:,b1]), list(X[:,b2]), c = y, cmap=cmap, alpha=0.8)
ax2.set_title("Actual")
ax1.set_xlabel("Band {}".format(b1+1))
ax1.set_ylabel("Band {}".format(b2+1))

nosnow_patch = patches.Patch(color=cmap(0.999999), label="Snow On")
snow_patch = patches.Patch(color=cmap(0), label='Snow Off')

#ax1.legend(handles=[nosnow_patch, snow_patch])
ax2.legend(handles=[nosnow_patch, snow_patch])

# incorrect = abs(preds - merged.label)
# ax3.set_title("Mistakes")
# ax3.scatter(merged[b1], merged[b2], color=colors[incorrect], alpha=0.9)
# right_patch = patches.Patch(color=colors[0], label="Correct")
# wrong_patch = patches.Patch(color=colors[1], label='Incorrect')

# ax1.legend(handles=[nosnow_patch, snow_patch])
# ax2.legend(handles=[nosnow_patch, snow_patch])
# ax3.legend(handles=[right_patch, wrong_patch])

fig.suptitle("GPC Probabilistic Performance")
#plt.tight_layout(w_pad=0.9)




# which bands to plot?

actualPreds = model.predict(X)
accuracy = accuracy_score(y, actualPreds)

cmap = plt.get_cmap("coolwarm_r")

binary_colors = np.array([cmap(0), cmap(1)])

fig, (ax1, ax2) = plt.subplots(1, 2)
s1 = ax1.scatter(X[:,b1], X[:,b2], c = preds, cmap=cmap, alpha=0.8, vmin=0, vmax=1)
ax1.set_title("Predictions (Threshold)")
ax1.set_xlabel(b1)
ax1.set_ylabel(b2)
#plt.colorbar(s1, ax=ax1, orientation='horizontal')

ax2.scatter(X[:,b1], X[:,b2], c = y, cmap=cmap, alpha=0.8)
ax2.set_title("Actual")
ax2.set_xlabel(b1)
ax2.set_ylabel(b2)

nosnow_patch = patches.Patch(color=cmap(0.999999), label="Snow On")
snow_patch = patches.Patch(color=cmap(0), label='Snow Off')

ax1.legend(handles=[nosnow_patch, snow_patch])
ax2.legend(handles=[nosnow_patch, snow_patch])

# incorrect = abs(preds - merged.label)
# ax3.set_title("Mistakes")
# ax3.scatter(merged[b1], merged[b2], color=colors[incorrect], alpha=0.9)
# right_patch = patches.Patch(color=colors[0], label="Correct")
# wrong_patch = patches.Patch(color=colors[1], label='Incorrect')

# ax1.legend(handles=[nosnow_patch, snow_patch])
# ax2.legend(handles=[nosnow_patch, snow_patch])
# ax3.legend(handles=[right_patch, wrong_patch])

fig.suptitle("GPC Threshold Performance (acc: {:f})".format(accuracy * 100))

#plt.tight_layout(w_pad=0.9)


rand_indices = np.random.randint(0, len(X)-1, 5000)
thosedata = data.iloc[rand_indices]
X = np.array(thosedata[bands])
y = np.array(thosedata.label)
X = StandardScaler().fit_transform(X)
_, snowOnProbs = zip(*model.predict_proba(X))

# which bands to plot?
b1 = 1
b2 = 3
preds = snowOnProbs

cmap = plt.get_cmap("coolwarm_r")

binary_colors = np.array([cmap(0), cmap(1)])

fig, axes = plt.subplots(1, 2)
ax1, ax2 = axes
s1 = ax1.scatter(list(X[:,b1]), list(X[:,b2]), c = preds, cmap=cmap, alpha=0.8, vmin=0, vmax=1)
ax1.set_title("Predictions (Probability)")
ax1.set_xlabel(b1)
ax1.set_ylabel(b2)
fig.colorbar(s1, orientation='horizontal', ax=axes.ravel().tolist())

ax2.scatter(list(X[:,b1]), list(X[:,b2]), c = y, cmap=cmap, alpha=0.8)
ax2.set_title("Actual")
ax2.set_xlabel(b1)
ax2.set_ylabel(b2)

nosnow_patch = patches.Patch(color=cmap(0.999999), label="Snow On")
snow_patch = patches.Patch(color=cmap(0), label='Snow Off')

#ax1.legend(handles=[nosnow_patch, snow_patch])
ax2.legend(handles=[nosnow_patch, snow_patch])

# incorrect = abs(preds - merged.label)
# ax3.set_title("Mistakes")
# ax3.scatter(merged[b1], merged[b2], color=colors[incorrect], alpha=0.9)
# right_patch = patches.Patch(color=colors[0], label="Correct")
# wrong_patch = patches.Patch(color=colors[1], label='Incorrect')

# ax1.legend(handles=[nosnow_patch, snow_patch])
# ax2.legend(handles=[nosnow_patch, snow_patch])
# ax3.legend(handles=[right_patch, wrong_patch])

fig.suptitle("GPC Probabilistic Performance")
#plt.tight_layout(w_pad=0.9)




