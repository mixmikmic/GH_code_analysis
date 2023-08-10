get_ipython().run_line_magic('run', '__init__.py')
get_ipython().run_line_magic('matplotlib', 'inline')

db_1 = pd.read_pickle('./Datasets/database_1.p')
db_2 = pd.read_pickle('./Datasets/database_2.p')
db_3 = pd.read_pickle('./Datasets/database_3.p')
# top 20 features found 
db_top_20 =  ['feat_257', 'feat_269', 'feat_308', 'feat_315', 'feat_336', 'feat_341', 
                   'feat_395', 'feat_504', 'feat_526', 'feat_639', 'feat_681', 'feat_701', 
                   'feat_724', 'feat_736', 'feat_769', 'feat_808', 'feat_829', 'feat_867',
                   'feat_920', 'feat_956']

#create X and y dataframes from samplesets 
db_y_1 = db_1['target']
db_x_1 = db_1[db_top_20]
db_y_2 = db_2['target']
db_x_2 = db_2[db_top_20]
db_y_3 = db_3['target']
db_x_3 = db_3[db_top_20]

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
db_1 = skb_5_feats(db_x_1, db_y_1)
db_2 = skb_5_feats(db_x_2, db_y_2)
db_3 = skb_5_feats(db_x_3, db_y_3)

print(np.sort(db_1))
print(np.sort(db_2))
print(np.sort(db_3))

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

db_1_rfe = rfe_5_feats(db_x_1, db_y_1)
db_2_rfe = rfe_5_feats(db_x_2, db_y_2)
db_3_rfe = rfe_5_feats(db_x_3, db_y_3)

print(np.sort(db_1_rfe))
print(np.sort(db_2_rfe))
print(np.sort(db_3_rfe))

from sklearn.pipeline import Pipeline

def feature_importance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    
    
    rf_pipe = Pipeline([('scaler',StandardScaler()),
    ('clf',RandomForestClassifier(random_state=42))])
    
    rfparams = {
    'clf__n_estimators':[10,50],
    'clf__max_features':['auto','log2']}
    
    rfgs = GridSearchCV(rf_pipe, rfparams, cv=5, n_jobs=-1)
    
    rfgs.fit(X_train, y_train)
    
    important_features = rfgs.best_estimator_.named_steps['clf']
    
    return important_features

db_1 = feature_importance(db_x_1, db_y_1)

db_1

db_1 = feature_importance(db_x_1, db_y_1)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(db_x_1.columns, db_1.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances.sort_values(['Gini-importance'], ascending=False).head(5)

db_2 = feature_importance(db_x_2, db_y_2)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(db_x_2.columns, db_2.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances_2 = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances_2.sort_values(['Gini-importance'], ascending=False).head(5)

db_3 = feature_importance(db_x_3, db_y_3)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(db_x_3.columns, db_3.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances_3 = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances_3.sort_values(['Gini-importance'], ascending=False).head(5)



