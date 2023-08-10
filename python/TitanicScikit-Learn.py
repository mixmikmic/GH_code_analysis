from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding, BaggingClassifier
from sklearn.svm import SVC, LinearSVC, LinearSVR, NuSVC, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import sys

def modify_data(base_df):
    new_df = pd.DataFrame()
    new_df['Gender'] = base_df.Sex.map(lambda x:1 if x.lower() == 'female' else 0)
    fares_by_class = base_df.groupby('Pclass').Fare.median()

    def getFare(example):
        if pd.isnull(example):
            example['Fare'] = fares_by_class[example['Pclass']]
        return example
    new_df['Fare'] = base_df['Fare']

    new_df['Family'] = (base_df.Parch + base_df.SibSp) > 0
    new_df['Family'] = new_df['Family'].map(lambda x:1 if x else 0)
    new_df['GenderFam'] = new_df['Gender']+new_df['Family']
    new_df['Title'] = base_df.Name.map(lambda x:x.split(' ')[0])
    new_df['Rich'] = base_df.Pclass == 1

    return new_df

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

y = train.Survived.values
ids = test['PassengerId'].values

train = modify_data(train)
test = modify_data(test)

train = train.fillna(-1)
test = test.fillna(-1)

for f in train.columns:
    if train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

models = {'xgb': xgb.XGBClassifier(n_estimators=2700,
                                   nthread=-1,
                                   max_depth=12,
                                   learning_rate=0.09,
                                   silent=True,
                                   subsample=0.8,
                                   colsample_bytree=0.75),
          'rf': RandomForestClassifier(n_estimators = 150, criterion='gini'),
          'linearsvc': LinearSVC(C=0.13, loss='hinge'),
          'linearsvr': LinearSVR(),
          'nusvc': NuSVC(),
          'nusvr': NuSVR(),
          'dtc': DecisionTreeClassifier(),
          'dtr': DecisionTreeRegressor(),
          'etc': ExtraTreeClassifier(),
          'etr': ExtraTreeRegressor(),
          'rfr': RandomForestRegressor(),
          'bc': BaggingClassifier(),
          'lr': LinearRegression(),
          'logit': LogisticRegression()}

model_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, train, y, cv=3)
    model_scores[name] = scores.mean()
    print(name, model_scores[name])

pred_array = []
for m, score in model_scores.items():
    if score < 0.76:
        continue
    model = models[m].fit(train, y)
    preds = model.predict(test)
    pred_array.append(preds)
    results = pd.DataFrame({"PassengerId":ids, 'Survived': preds})
    results['PassengerId'] = results['PassengerId'].astype('int')
    results.set_index("PassengerId")
    results.to_csv('output/test_results_{}.csv'.format(m), index=False)

ensemble_preds = [0]*len(ids)
for p in pred_array:
    if not ensemble_preds:
        ensemble_preds = p
    else:
        ensemble_preds = [a+b for a, b in zip(ensemble_preds, p)]

votes = [0 if a < len(pred_array)/2 else 1 for a in ensemble_preds]
results = pd.DataFrame({"PassengerId":ids, 'Survived': votes})
results['PassengerId'] = results['PassengerId'].astype('int')
results.set_index("PassengerId")
results.to_csv('output/test_results_ensemble.csv'.format(m), index=False)

