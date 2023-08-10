import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

plt.rc('font', size=14)        # controls default text sizes
plt.rc('axes', titlesize=16)   # fontsize of the axes title
plt.rc('axes', labelsize=16)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.rc('legend', fontsize=14)  # legend fontsize

f = open('databall.pkl')
X, X_train, X_test, y, y_train, y_test = pickle.load(f)
f.close()

def evaluate_model(model, X, y, k=6):
    # Define metrics
    scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'average_precision']
    
    # Create cross validator
    kfold = StratifiedKFold(n_splits=k)
    
    # Calculate metrics
    return [cross_val_score(model, X, y, cv=kfold, scoring=score).mean() for score in scoring]

def get_metrics(models, attributes, param_name, param_vec):
    # Initialize list of results
    results = [[] for m in models]
    
    # Make transformer that selects the desired attributes from the DataFrame
    selector = FunctionTransformer(partial(select_columns, names=attributes))
    
    for i in range(len(models)):
        for param in param_vec:
            # Make a pipeline that selects the desired attributes prior to the classifier
            model = make_pipeline(selector, models[i](**{param_name: param}))
            
            # Calculate performance metrics
            results[i] += [evaluate_model(model, X_train, y_train)]
    
    return results

def plot_metrics(x, y, xlabel, legend, legendsize=14, figsize=(16, 8), log=False):
    rows = 2
    cols = 3
    ylabel = ['Accuracy', 'Precision', 'Recall', 'ROC Area', 'Precision/Recall Area']
    plt.figure(figsize=figsize)
    
    for i in range(0,len(y[0][0])):
        ax = plt.subplot(100*rows + 10*cols + i + 1)
        
        if log:
            [ax.semilogx(x, [yvec[i] for yvec in y[j]]) for j in range(0,len(y))]
        else:
            [ax.plot(x, [yvec[i] for yvec in y[j]]) for j in range(0,len(y))]
            ax.set_xlim(0)
        
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel[i])
        ax.legend(legend, fontsize=legendsize)
    
    plt.tight_layout()
    plt.show()

def select_columns(df, names, columns=X_train.columns):
    return df[:, [i for i, col in enumerate(columns) if any(n in col for n in names)]]

# Create functions that return logistic regression models with different parameters
models = [partial(LogisticRegression, penalty='l1'),
          partial(LogisticRegression, penalty='l1', class_weight='balanced'),
          partial(LogisticRegression),
          partial(LogisticRegression, class_weight='balanced'),
          partial(LogisticRegression, dual=True),
          partial(LogisticRegression, class_weight='balanced', dual=True)]

C_vec = np.logspace(start=-3, stop=3, num=10)
results = get_metrics(models, ['SRS'], 'C', C_vec)
legend = ['L1 Norm', 'L1 Norm, Balanced Class', 'L2 Norm (Default)',
          'L2 Norm, Balanced Class', 'L2 Norm, Dual Form', 'L2 Norm, Balanced Class, Dual Form']
plot_metrics(C_vec, results, 'Regularization Parameter', legend, log=True)

# Create functions that return support vector machines with different parameters
models = [partial(LinearSVC, penalty='l1', dual=False),
          partial(LinearSVC, penalty='l1', dual=False, class_weight='balanced'),
          partial(LinearSVC, loss='hinge'),
          partial(LinearSVC, loss='hinge', class_weight='balanced'),
          partial(LinearSVC),
          partial(LinearSVC, dual=False),
          partial(LinearSVC, class_weight='balanced'),
          partial(LinearSVC, dual=False, class_weight='balanced')]

C_vec = np.logspace(start=-3, stop=3, num=10)
results = get_metrics(models, ['SRS'], 'C', C_vec)
legend = ['L1 Norm, Squared Hinge', 'L1 Norm, Squared Hinge, Balanced', 'L2 Norm, Hinge, Dual Form',
          'L2 Norm, Hinge, Dual Form, Balanced', 'L2 Norm, Squared Hinge, Dual Form (Default)',
          'L2 Norm, Squared Hinge', 'L2 Norm, Squared Hinge, Dual Form, Balanced',
          'L2 Norm, Squared Hinge, Balanced']
plot_metrics(C_vec, results, 'Regularization Parameter', legend, legendsize=12, log=True)

# Create functions that return random forest models with different parameters
models = [partial(RandomForestClassifier),
          partial(RandomForestClassifier, class_weight='balanced'),
          partial(RandomForestClassifier, criterion='entropy'),
          partial(RandomForestClassifier, criterion='entropy', class_weight='balanced')]

n_vec = range(2, 21, 2)
results = get_metrics(models, ['SRS'], 'n_estimators', n_vec)
legend = ['Gini (Default)', 'Gini, Balanced Class', 'Entropy', 'Entropy, Balanced Class']
plot_metrics(n_vec, results, 'Forest Size', legend)

# Create functions that return neural networks with different parameters
models = [partial(MLPClassifier, hidden_layer_sizes=5),
          partial(MLPClassifier, hidden_layer_sizes=10),
          partial(MLPClassifier),
          partial(MLPClassifier, activation='logistic', hidden_layer_sizes=5),
          partial(MLPClassifier, activation='logistic', hidden_layer_sizes=10),
          partial(MLPClassifier, activation='logistic')]

alpha_vec = np.logspace(start=-3, stop=3, num=10)
results = get_metrics(models, ['SRS'], 'alpha', alpha_vec)
legend = ['RELU w/ 5 Hidden Neurons', 'RELU w/ 10 Hidden Neurons', 'RELU w/ 100 Hidden Neurons (Default)',
          'Logistic w/ 5 Hidden Neurons', 'Logistic w/ 10 Hidden Neurons', 'Logistic w/ 100 Hidden Neurons']
plot_metrics(alpha_vec, results, 'Regularization Parameter', legend, log=True)

