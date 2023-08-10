get_ipython().system('sudo pip install xlrd')

import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='ticks', palette='Set2')

# some custom libraries!
import sys
sys.path.append("..")
from ds_utils.decision_surface import *

concrete_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

concrete_df = pd.read_excel(concrete_url).dropna()
concrete_df.head(5)

concrete_df[["Concrete compressive strength(MPa, megapascals) "]].hist()
plt.show()

compression_median = concrete_df["Concrete compressive strength(MPa, megapascals) "].median()
concrete_df["class"] = concrete_df["Concrete compressive strength(MPa, megapascals) "].apply(
        lambda strength: 1 if strength > compression_median else 0
    )
concrete_df.columns

concrete_df.rename(columns={
    "Cement (component 1)(kg in a m^3 mixture)"             : "Cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)" : "Slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)"            : "Fly Ash",
    "Water  (component 4)(kg in a m^3 mixture)"             : "Water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)"   : "Superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)"  : "Coarse Agg",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)"     : "Fine Agg",
    "Age (day)"                                             : "Age",
    "Concrete compressive strength(MPa, megapascals) "      : "Strength"
}, inplace= True)

predictor_columns = [c for c in concrete_df.columns if c != "Strength" and c != "class"]
predictor_columns

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def score_model(model, df, predictor_columns, class_column, scoring="accuracy"):
    scores = cross_val_score(model, df[predictor_columns], df[class_column], scoring=scoring)
    return {"mean": scores.mean(), "std_dev": scores.std()}

depths = list(range(2, 50, 3))
scores_list = [score_model(DecisionTreeClassifier(max_depth=depth),
                           concrete_df,
                           predictor_columns,
                           "class")
               for depth in depths]

accys = np.array([score["mean"] for score in scores_list])
accys_std = np.array([score["std_dev"] for score in scores_list])

plt.plot(depths, accys)
plt.fill_between(depths, accys + accys_std, accys - accys_std, alpha=0.3)
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Tuning Tree Depth using Cross Validation")
plt.show()

from sklearn.model_selection import GridSearchCV

grid = {
    "max_depth": list(range(2, 53, 5)),
    "min_samples_leaf": list(range(2, 53, 5))
}

# gridsearchcv behaves just like a model, with fit and predict, with some additional
# functionality too
tuned_model = GridSearchCV(DecisionTreeClassifier(), grid, scoring="accuracy")
tuned_model.fit(concrete_df[predictor_columns], concrete_df["class"])

print ("Best accuracy: %0.3f, using: " % tuned_model.best_score_)
print (tuned_model.best_params_)

from mpl_toolkits.mplot3d import Axes3D

means = tuned_model.cv_results_['mean_test_score']
stds = tuned_model.cv_results_['std_test_score']
params = tuned_model.cv_results_['params']

min_samples_leafs = [param["min_samples_leaf"] for param in params]
max_depths = [param["max_depth"] for param in params]

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(min_samples_leafs, max_depths, means, linewidth=0.1)
ax.set_xlabel('min samples in a leaf')
ax.set_ylabel('maximum tree depth')
ax.set_zlabel('accuracy')

plt.show()

from sklearn.ensemble import RandomForestClassifier

grid = {
    "n_estimators": list(range(1, 100, 10)),
    "min_samples_leaf": list(range(2, 200, 20)),
    "criterion": ["entropy", "gini"]
}

# increased verbosity
rf_tuned_model = GridSearchCV(RandomForestClassifier(), grid, scoring="accuracy", verbose=1)
rf_tuned_model.fit(concrete_df[predictor_columns], concrete_df["class"])

print ("Best accuracy: %0.3f, using: " % rf_tuned_model.best_score_)
print (rf_tuned_model.best_params_)

from sklearn.linear_model import LogisticRegression

# Plot different regularization values for L1 and L2 regularization
for regularization in ['l2', 'l1']:
    
    # Print what we are doing
    print ("\nFitting with %s regularization: \n" % regularization)
    position = 0

    plt.figure(figsize=[15, 21])
    c_values = [np.power(10.0, c) for c in range(-6, 3)]
    for c in c_values:
        position += 1
        plt.subplot(3, 3, position)
        
        model = LogisticRegression(penalty=regularization, C=c)
        Decision_Surface(concrete_df[predictor_columns],
                         "Cement",
                         "Slag",
                         concrete_df["class"],
                         model,
                         probabilities=True,
                         sample=1)
        plt.title("C=%f" % c)
    plt.tight_layout()
    plt.show()

concrete_df.head(5)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
concrete_df[predictor_columns] = scaler.fit_transform(concrete_df[predictor_columns])

concrete_df.head(5)

def get_lr_coeffs(df, lr, predictor_columns, class_column):
    lr.fit(df[predictor_columns], df[class_column])

    return dict(zip(predictor_columns, lr.coef_[0]))

coefs = get_lr_coeffs(concrete_df, LogisticRegression(), predictor_columns, "class")
pd.DataFrame([coefs])

def get_lr_regularization_paths(df, predictor_columns, class_column, regtype, reg_values):
    coefs = [get_lr_coeffs(concrete_df,
                          LogisticRegression(penalty=regtype, C=10**reg),
                          predictor_columns,
                          class_column)
             for reg in reg_values]

    df = pd.DataFrame(coefs)
    df["regularization"] = reg_values
    
    df.set_index("regularization", inplace=True)
    
    return df

regs = np.arange(-5, 5, 0.5)  #go through a bunch of ascending regularization parameters
    
l1_coefs = get_lr_regularization_paths(concrete_df,
                                       predictor_columns,
                                       "class",
                                       "l1",
                                       regs)

l2_coefs = get_lr_regularization_paths(concrete_df,
                                       predictor_columns,
                                       "class",
                                       "l2",
                                       regs)

l1_coefs.plot()
plt.title("L1 Regularization paths")

l2_coefs.plot()
plt.title("L2 Regularization paths")

plt.show()

import sklearn.model_selection as cv
from sklearn.metrics import accuracy_score


def evaluate_model_on_sample(df, model, predictor_cols, class_col, pct, scoring=accuracy_score):
    kf = cv.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    X = df[predictor_cols]
    y = df[class_col]
    
    for train_index, test_index in kf.split(X, y):  
        # only take a portion of the training
        sampled_indices = np.random.permutation(range(len(train_index)))[:int(pct*len(train_index))].tolist()
        np_train = np.array(train_index)
        to_get = np_train[sampled_indices]

        model.fit(X.loc[to_get], y[to_get])
        scores.append(scoring(y[test_index], model.predict(X.loc[test_index])))
        
    return np.mean(scores), np.std(scores)

pcts = np.linspace(0.01,1,100).tolist()
dt_scores = [evaluate_model_on_sample(concrete_df,
                                     DecisionTreeClassifier(),
                                     predictor_columns,
                                     "class",
                                     pct)
             for pct in pcts]

lr_scores = [evaluate_model_on_sample(concrete_df,
                                      LogisticRegression(),
                                      predictor_columns,
                                      "class",
                                      pct)
             for pct in pcts]

raw_dt_score = np.array([s[0] for s in dt_scores])
std_dt_score = np.array([s[1] for s in dt_scores])

raw_lr_score = np.array([s[0] for s in lr_scores])
std_lr_score = np.array([s[1] for s in lr_scores])



plt.plot(pcts, raw_dt_score, label="Classifier Tree")
plt.plot(pcts, raw_lr_score, label="Logistic Regression")
plt.xlabel("Percent of data")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(pcts, raw_dt_score, label="Classifier Tree")
plt.fill_between(pcts, raw_dt_score + std_dt_score, raw_dt_score - std_dt_score, alpha=0.3)
plt.plot(pcts, raw_lr_score, label="Logistic Regression")
plt.fill_between(pcts, raw_lr_score + std_lr_score, raw_lr_score - std_lr_score, alpha=0.3)
plt.xlabel("Percent of data")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



