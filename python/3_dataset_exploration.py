import pandas as pd
startups = pd.read_csv('data/startups_2.csv', index_col=0)
startups[:3]

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def plot_avg_status_against_avg_total(df, status):
    startups_numeric = df.filter(regex=('(number_of|avg_).*|.*(funding_total_usd|funding_rounds|_at|status)'))
    startups_acquired = startups_numeric[startups_numeric['status'] == status]

    startups_numeric = startups_numeric.drop('status', 1)
    startups_acquired = startups_acquired.drop('status', 1)

    fig, ax = plt.subplots(figsize=(20,20)) 
    ax.set_title(status+' startups heatmap')
    sns.heatmap((pd.DataFrame(startups_acquired.mean()).transpose() -startups_numeric.mean())/startups_numeric.std(ddof=0), annot=True, cbar=False, square=True, ax=ax)

plot_avg_status_against_avg_total(startups, 'acquired')

plot_avg_status_against_avg_total(startups, 'closed')

plot_avg_status_against_avg_total(startups, 'ipo')

plot_avg_status_against_avg_total(startups, 'operating')

# Produce a scatter matrix for each pair of features in the data
#startups_funding_rounds = startups_numeric.filter(regex=('.*funding_total_usd'))
#pd.scatter_matrix(startups_funding_rounds, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

from sklearn.decomposition import PCA
import visuals as vs


startups_numeric = startups.filter(regex=('(number_of|avg_).*|.*(funding_total_usd|funding_rounds|_at)'))

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=4)
pca.fit(startups_numeric)


# Generate PCA results plot
pca_results = vs.pca_results(startups_numeric, pca)
startups_numeric[:3]

good_data = startups_numeric
import numpy as np
dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
components.index = dimensions
components

startups_numeric_acquired = startups.filter(regex=('(number_of|avg_).*|.*(funding_total_usd|funding_rounds|_at|status)'))
startups_numeric_acquired = startups_numeric_acquired[startups_numeric_acquired['status'] == 'acquired']
startups_numeric_acquired = startups_numeric_acquired.drop('status', 1)

pca = PCA(n_components=4)
pca.fit(startups_numeric_acquired)

# Generate PCA results plot
pca_results = vs.pca_results(startups_numeric_acquired, pca)

#startups_numeric = df.filter(regex=('.*(funding_total_usd|funding_rounds|status)'))
startups_non_numeric = startups.filter(regex=('^((?!(_acquisitions|_investments|_per_round|funding_total_usd|funding_rounds|_at)).)*$'))
startups_non_numeric[:3]

startups_non_numeric['status'].value_counts()
startups_non_numeric['acquired'] = startups_non_numeric['status'].map({'operating': 0, 'acquired':1, 'closed':0, 'ipo':0})
startups_non_numeric = startups_non_numeric.drop('status', 1)
startups_non_numeric[:3]

from sklearn import tree
def visualize_tree(tree_model, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree_model -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        tree.export_graphviz(tree_model, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")



#import visuals_tree as vs_tree
#vs_tree.ModelLearning(startups_non_numeric.drop(['acquired','state_code'], 1), startups_non_numeric['acquired'])
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from sklearn import grid_search
from sklearn import preprocessing


#clf = tree.DecisionTreeClassifier(random_state=0)
#cross_val_score(clf, startups_non_numeric.drop(['acquired','state_code'], 1),  startups_non_numeric['acquired'], cv=10)


#Drop state_code feature
features = startups_non_numeric.drop(['acquired','state_code'], 1)

#Convert state_code feature to number
#features = startups_non_numeric.drop(['acquired'], 1)
#features['state_code'] = preprocessing.LabelEncoder().fit_transform(features['state_code'])

#Convert state_code to dummy variables
features = pd.get_dummies(startups_non_numeric.drop(['acquired'], 1), prefix='state', columns=['state_code'])


#Merge numeric_features to non-numeric-features
features_all = pd.concat([features, startups_numeric], axis=1, ignore_index=False)
#features = features_all

features = startups_numeric






parameters = {'max_depth':range(5,20)}
clf = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=5, scoring='roc_auc')
clf.fit(X=features, y=startups_non_numeric['acquired'])
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 
print tree.export_graphviz(clf.best_estimator_, feature_names=list(features.columns))

import visuals_tree as vs_tree
vs_tree = reload(vs_tree)
vs_tree.ModelComplexity(features_all, startups_non_numeric['acquired'])

all = pd.concat([features_all, startups_non_numeric['acquired']], axis=1, ignore_index=False)
all.to_csv('data/startups_3.csv')

all_with_status = all.join(startups['status'])
all_with_status_without_operating = all_with_status[all_with_status['status'] != 'operating']
all_with_status_without_operating.shape
all_without_operating = all_with_status_without_operating.drop('status', 1)
all_without_operating.to_csv('data/startups_not_operating_3.csv')

