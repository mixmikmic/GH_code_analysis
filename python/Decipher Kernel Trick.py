get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from sklearn.datasets import make_moons, make_circles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

n_samples = 10000

#circles = make_circles(n_samples=n_samples, factor=.5)
#moons = make_moons(n_samples=n_samples)
noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = make_moons(n_samples=n_samples, noise=.05)
#no_structure = np.random.rand(n_samples, 2), None
#datasets = [noisy_circles, noisy_moons,no_structure]

datasets = [noisy_circles, noisy_moons]

from sklearn import svm


xx = np.linspace(-1,2)

for data in datasets:
    X, y = data
    
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    df = pd.DataFrame(X, columns = ('x', 'y'))
    df['class'] = y

    groups = df.groupby('class')

    # Plot
    plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
    colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')

    fig, ax = plt.subplots()
    ax.set_color_cycle(colors)
    #ax.set_prop_cycle(colors)
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='.', linestyle='', ms=12, label=name)
    ax.legend(numpoints=1, loc='upper left')

    #plt.show()
    
    # train a linear SVM
    linear_svm = svm.SVC(kernel='linear', C = 1.0)
    linear_svm.fit(X_train,y_train)
    y_pred = linear_svm.predict(X_test)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)

    #intuituion behind line plotting is given in next cell
    w = linear_svm.coef_[0]
    #print(w)
    a = -w[0] / w[1]
    #xx = np.linspace(0,12)
    yy = a * xx - linear_svm.intercept_[0] / w[1]
    h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

    #plt.scatter(X[:, 0], X[:, 1], c = y)
    #plt.legend()
    
    
    print "LINEAR SVM"
    plt.show()
    print '\t', 'precision','\trecall', '\t\tf1-score','\tsupport'
    print "%d \t%0.2f \t\t%0.2f \t\t%0.2f \t\t%0.2f" %(0, precision[0], recall[0], fbeta_score[0], support[0])
    print "%d \t%0.2f \t\t%0.2f \t\t%0.2f \t\t%0.2f" %(1, precision[1], recall[1], fbeta_score[1], support[1])

    

from sklearn import svm


xx = np.linspace(-1,2)

for data in datasets:
    X, y = data
    
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    df = pd.DataFrame(X, columns = ('x', 'y'))
    df['class'] = y

    groups = df.groupby('class')

    # Plot
    plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
    colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')

    fig, ax = plt.subplots()
    ax.set_color_cycle(colors)
    #ax.set_prop_cycle(colors)
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='.', linestyle='', ms=12, label=name)
    ax.legend(numpoints=1, loc='upper left')

    #plt.show()
    
    # train a non-linear SVM
    clf = svm.NuSVC()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)

    # plot the decison function
    # plot the decision function for each datapoint on the grid - intuition behind the plot is given below
    xx, yy = np.meshgrid(np.linspace(-2, 2, 500),np.linspace(-2, 2, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',           origin='lower')
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,linetypes='--')

    #plt.scatter(X[:, 0], X[:, 1], c = y)
    #plt.legend()
    
    
    print "Non-Linear SVM with RBF"
    plt.show()
    print '\t', 'precision','\trecall', '\t\tf1-score','\tsupport'
    print "%d \t%0.2f \t\t%0.2f \t\t%0.2f \t\t%0.2f" %(0, precision[0], recall[0], fbeta_score[0], support[0])
    print "%d \t%0.2f \t\t%0.2f \t\t%0.2f \t\t%0.2f" %(1, precision[1], recall[1], fbeta_score[1], support[1])

    

print clf.decision_function_shape

clf.coef_



