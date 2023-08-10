# Libraries to be used
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from lime.lime_tabular import LimeTabularExplainer
from scipy.sparse import hstack

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Load data set for experiments
from sklearn.datasets import load_wine
dataset = load_wine()
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
y = dataset.target

# Split data into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    stratify = y,
    train_size = 100,
    random_state = 2018   # for reproducibility
)

# Take a glance at feature distributions, broken down by class
Xtrain.groupby(ytrain).quantile([.1,.9])

# Train a random forest classifier for each of the three classes
clf = []
for clas in range(3):
    clf.append(
        RandomForestClassifier(
            n_estimators = 100, n_jobs = -1,
            random_state = 2018   # for reproducibility
        ).fit(Xtrain, ytrain == clas)
    )

# Check the AUC's of the classifiers on the test data
for clas in range(3):
    print(roc_auc_score(ytest == clas, clf[clas].predict_proba(Xtest)[:,1]))

# Create a LIME explainer for tabular data
explainer = LimeTabularExplainer(
    Xtrain.values, feature_names = Xtrain.columns,
    random_state = 2018   # for reproducibility
)

def explain_row(clf, row, num_reasons = 2):
    '''
    Produce LIME explanations for a single row of data.
        * `clf` is a binary classifier (with a predict_proba method),
        * `row` is a row of features data,
        * `num_reasons` (default 2) is the number of 
          reasons/explanations to be produced.
          
    '''
    exp = [
        exp_pair[0] for exp_pair in     # Get each explanation (a string)
        explainer.explain_instance(     # from the LIME explainer
            row, clf.predict_proba,     # for the given row and classifier
            labels = [1],               # and label 1 ("positives")
            num_features = num_reasons  # for up to `num_reasons` explanations
        ).as_list()
        if exp_pair[1] > 0              # but only for positive explanations 
    ][:num_reasons]
    
    # Fill in any missing explanations with blanks
    exp += [''] * (num_reasons - len(exp))  
    return exp


def predict_explain(rf, X, num_reasons = 2):
    '''
    Produce scores and LIME explanations for every row in a data frame.
        * `rf` is a binary classifier with a predict_proba method,
        * `X` is the features data frame,
        * `num_reasons` (default 2) is the number of 
          reasons/explanations to be produced for each row.
          
    '''
    # Prepare the structure to be returned
    pred_ex = X[[]]
    
    # Get the scores from the classifier
    pred_ex['SCORE'] = rf.predict_proba(X)[:,1]
    
    # Get the reasons/explanations for each row
    cols = zip(
        *X.apply(
            lambda x: explain_row(rf, x, num_reasons), 
            axis = 1, raw = True
        )
    )
    
    # Return the results
    for n in range(num_reasons):
        pred_ex['REASON%d' % (n+1)] = next(cols)
    return pred_ex

get_ipython().run_cell_magic('time', '', "pe0l = predict_explain(clf[0], Xtest).assign(\n    TRUE_CLASS = ytest\n).sort_values('SCORE', ascending = False).head(20)")

pe0l

get_ipython().run_cell_magic('time', '', "pe1l = predict_explain(clf[1], Xtest).assign(\n    TRUE_CLASS = ytest\n).sort_values('SCORE', ascending = False).head(20)")

pe1l

get_ipython().run_cell_magic('time', '', "pe2l = predict_explain(clf[2], Xtest).assign(\n    TRUE_CLASS = ytest\n).sort_values('SCORE', ascending = False).head(20)")

pe2l

import tree_explainer

get_ipython().run_cell_magic('time', '', "pe0t = tree_explainer.predict_explain(clf[0], Xtest).assign(\n    TRUE_CLASS = ytest\n).sort_values('SCORE', ascending = False).head(20)")

pe0t

get_ipython().run_cell_magic('time', '', "pe1t = tree_explainer.predict_explain(clf[1], Xtest).assign(\n    TRUE_CLASS = ytest\n).sort_values('SCORE', ascending = False).head(20)")

pe1t

get_ipython().run_cell_magic('time', '', "pe2t = tree_explainer.predict_explain(clf[2], Xtest).assign(\n    TRUE_CLASS = ytest\n).sort_values('SCORE', ascending = False).head(20)")

pe2t

pe0l[['REASON1','REASON2']].join(
    pe0t[['REASON1','REASON2']], 
    lsuffix = '_LIME'
)

pe1l[['REASON1','REASON2']].join(
    pe1t[['REASON1','REASON2']], 
    lsuffix = '_LIME'
)

pe2l[['REASON1','REASON2']].join(
    pe2t[['REASON1','REASON2']], 
    lsuffix = '_LIME'
)

