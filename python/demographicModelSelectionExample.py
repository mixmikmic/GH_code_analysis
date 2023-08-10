#untar and compile ms and sample_stats
get_ipython().system('tar zxf ms.tar.gz; cd msdir; gcc -o ms ms.c streec.c rand1.c -lm; gcc -o sample_stats sample_stats.c tajd.c -lm')
#I get three compiler warnings from ms, but everything should be fine
#now I'll just move the programs into the current working dir
get_ipython().system('mv msdir/ms . ; mv msdir/sample_stats .;')

get_ipython().system('conda install scikit-learn --yes')

get_ipython().system('pip install -U scikit-learn')

#simulate under the equilibrium model
get_ipython().system('./ms 20 2000 -t 100 -r 100 10000 | ./sample_stats > equilibrium.msOut.stats')

#simulate under the contraction model
get_ipython().system('./ms 20 2000 -t 100 -r 100 10000 -en 0 1 0.5 -en 0.2 1 1 | ./sample_stats > contraction.msOut.stats')

#simulate under the growth model
get_ipython().system('./ms 20 2000 -t 100 -r 100 10000 -en 0.2 1 0.5 | ./sample_stats > growth.msOut.stats')

#now lets suck up the data columns we want for each of these files, and create one big training set; we will use numpy for this
# note that we are only using two columns of the data- these correspond to segSites and Fay & Wu's H
import numpy as np
X1 = np.loadtxt("equilibrium.msOut.stats",usecols=(3,9))
X2 = np.loadtxt("contraction.msOut.stats",usecols=(3,9))
X3 = np.loadtxt("growth.msOut.stats",usecols=(3,9))
X = np.concatenate((X1,X2,X3))

#create associated 'labels' -- these will be the targets for training
y = [0]*len(X1) + [1]*len(X2) + [2]*len(X3)
Y = np.array(y)

#the last step in this process will be to shuffle the data, and then split it into a training set and a testing set
#the testing set will NOT be used during training, and will allow us to check how well the classifier is doing
#scikit-learn has a very convenient function for doing this shuffle and split operation
#
# will will keep out 10% of the data for testing

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)

from sklearn.ensemble import RandomForestClassifier

rfClf = RandomForestClassifier(n_estimators=100,n_jobs=10)
clf = rfClf.fit(X_train, Y_train)

from sklearn.preprocessing import normalize

#These two functions (taken from scikit-learn.org) plot the decision boundaries for a classifier.
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.05):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

#Let's do the plotting
import matplotlib.pyplot as plt
fig,ax= plt.subplots(1,1)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1, h=0.2)
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# plotting only a subset of our data to keep things from getting too cluttered
ax.scatter(X_test[:200, 0], X_test[:200, 1], c=Y_test[:200], cmap=plt.cm.coolwarm, edgecolors='k')
ax.set_xlabel(r"$\theta_{w}$", fontsize=14)
ax.set_ylabel(r"Fay and Wu's $H$", fontsize=14)
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Classifier decision surface", fontsize=14)
plt.show()

#here's the confusion matrix function
def makeConfusionMatrixHeatmap(data, title, trueClassOrderLs, predictedClassOrderLs, ax):
    data = np.array(data)
    data = normalize(data, axis=1, norm='l1')
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)

    for i in range(len(predictedClassOrderLs)):
        for j in reversed(range(len(trueClassOrderLs))):
            val = 100*data[j, i]
            if val > 50:
                c = '0.9'
            else:
                c = 'black'
            ax.text(i + 0.5, j + 0.5, '%.2f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=9)

    cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues, ax=ax)
    cbar.set_label("Fraction of simulations assigned to class", rotation=270, labelpad=20, fontsize=11)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.axis('tight')
    ax.set_title(title)

    #labels
    ax.set_xticklabels(predictedClassOrderLs, minor=False, fontsize=9, rotation=45)
    ax.set_yticklabels(reversed(trueClassOrderLs), minor=False, fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    
#now the actual work
#first get the predictions
preds=clf.predict(X_test)

counts=[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
for i in range(len(Y_test)):
    counts[Y_test[i]][preds[i]] += 1
counts.reverse()
classOrderLs=['equil','contraction','growth']

#now do the plotting
fig,ax= plt.subplots(1,1)
makeConfusionMatrixHeatmap(counts, "Confusion matrix", classOrderLs, classOrderLs, ax)
plt.show()

X1 = np.loadtxt("equilibrium.msOut.stats",usecols=(1,3,5,7,9))
X2 = np.loadtxt("contraction.msOut.stats",usecols=(1,3,5,7,9))
X3 = np.loadtxt("growth.msOut.stats",usecols=(1,3,5,7,9))
X = np.concatenate((X1,X2,X3))
#create associated 'labels' -- these will be the targets for training
y = [0]*len(X1) + [1]*len(X2) + [2]*len(X3)
Y = np.array(y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)
rfClf = RandomForestClassifier(n_estimators=100,n_jobs=10)
clf = rfClf.fit(X_train, Y_train)
preds=clf.predict(X_test)
counts=[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
for i in range(len(Y_test)):
    counts[Y_test[i]][preds[i]] += 1
counts.reverse()
fig,ax= plt.subplots(1,1)
makeConfusionMatrixHeatmap(counts, "Confusion matrix", classOrderLs, classOrderLs, ax)
plt.show()

