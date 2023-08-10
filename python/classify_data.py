from contrib.analysis.classify import Classifier
c = Classifier()

from argparse import Namespace
ns = Namespace()
ns.database = 'stoqs_september2013_t'
ns.classifier='Decision_Tree'
ns.inputs=['bbp700', 'fl700_uncorr']
ns.labels=['diatom', 'dino1', 'dino2', 'sediment']
ns.test_size=0.4
ns.train_size=0.4
ns.verbose=True
c.args = ns

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
X, y = c.loadLabeledData('Labeled Plankton', classes=('diatom', 'sediment'))
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=c.args.test_size, train_size=c.args.train_size)

get_ipython().magic('pylab inline')
import pylab as plt
from matplotlib.colors import ListedColormap
plt.rcParams['figure.figsize'] = (27, 3)

for i, (name, clf) in enumerate(c.classifiers.iteritems()):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1, len(c.classifiers) + 1, i + 1)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')



