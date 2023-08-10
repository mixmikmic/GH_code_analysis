get_ipython().run_line_magic('run', '../__init__.py')

uci_train = pd.read_pickle('../Datasets/train_clean.p')

# top 20 features found 
uci_20_feats =  [28, 48, 64, 105, 128, 153, 241, 281, 318, 336, 338, 378, 
                   433, 442, 451, 453, 455, 472, 475, 493]

#take 3 samples from madelon uci train dataset
uci_sample1 = uci_train.sample(440)
uci_sample2 = uci_train.sample(440)
uci_sample3 = uci_train.sample(440)

#create X and y dataframes from samplesets 
uci_y_1 = uci_sample1['target']
uci_x_1 = uci_sample1[uci_20_feats]
uci_y_2 = uci_sample2['target']
uci_x_2 = uci_sample2[uci_20_feats]
uci_y_3 = uci_sample3['target']
uci_x_3 = uci_sample3[uci_20_feats]

def skb_5_feats(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=.2, 
                                                    random_state=42)
    skb_list = []
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    skb = SelectKBest(k=5)
    skb.fit(X_train_scaled, y_train)
    
    skb_feats = x.columns[skb.get_support()]
    
    skb_list.append(skb_feats)
    
    return skb_list

# find top 5 features from each sample set
uci_1 = skb_5_feats(uci_x_1, uci_y_1)
uci_2 = skb_5_feats(uci_x_2, uci_y_2)
uci_3 = skb_5_feats(uci_x_3, uci_y_3)

print(np.sort(uci_1))
print(np.sort(uci_2))
print(np.sort(uci_3))

def rfe_5_feats(x, y, estimator = DecisionTreeClassifier(max_depth=10)):
    
    X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=.2, 
                                                    random_state=42)
    
    rfe_list = []
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rfe = RFE(estimator = estimator, n_features_to_select=5)
    rfe.fit(X_train_scaled, y_train)
    
    rfe_feats = x.columns[rfe.get_support()]
    rfe_list.append(rfe_feats)
    
    return rfe_list

uci_1_rfe = rfe_5_feats(uci_x_1, uci_y_1)
uci_2_rfe = rfe_5_feats(uci_x_2, uci_y_2)
uci_3_rfe = rfe_5_feats(uci_x_3, uci_y_3)

print(np.sort(uci_1_rfe))
print(np.sort(uci_2_rfe))
print(np.sort(uci_3_rfe))

from sklearn.pipeline import Pipeline

def feature_importance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    
    
    rf_pipe = Pipeline([('scaler',StandardScaler()),
    ('clf',RandomForestClassifier(random_state=42))])
    
    rfparams = {
    'clf__n_estimators':[10,50,100],
    'clf__max_features':['auto','log2']}
    
    rfgs = GridSearchCV(rf_pipe, rfparams, cv=5, n_jobs=-1)
    
    rfgs.fit(X_train, y_train)
    
    important_features = rfgs.best_estimator_.named_steps['clf']
    
    return important_features

uci_1 = feature_importance(uci_x_1, uci_y_1)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(uci_x_1.columns, uci_1.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances_1 = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances_1.sort_values(['Gini-importance'], ascending=False).head(5)

uci_2 = feature_importance(uci_x_2, uci_y_2)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(uci_x_2.columns, uci_2.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances_2 = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances_2.sort_values(['Gini-importance'], ascending=False).head(5)

uci_3 = feature_importance(uci_x_3, uci_y_3)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(uci_x_3.columns, uci_3.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances_3 = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances_3.sort_values(['Gini-importance'], ascending=False).head(5)

uci_y_1.to_pickle('./Datasets/uci_y_1')
uci_x_1.to_pickle('./Datasets/uci_x_1')
uci_y_2.to_pickle('./Datasets/uci_y_2')
uci_x_2.to_pickle('./Datasets/uci_x_2')
uci_y_3.to_pickle('./Datasets/uci_y_3')
uci_x_3.to_pickle('./Datasets/uci_x_3')

