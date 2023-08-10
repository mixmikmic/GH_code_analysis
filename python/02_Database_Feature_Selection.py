get_ipython().system('conda install --yes tqdm')

get_ipython().system('conda install psycopg2 --yes')

get_ipython().run_line_magic('run', '../__init__.py')
get_ipython().run_line_magic('matplotlib', 'inline')

database_1 = pd.read_pickle('../Datasets/database_1.p')
database_2 = pd.read_pickle('../Datasets/database_2.p')
database_3 = pd.read_pickle('../Datasets/database_3.p')

db_y_1 = database_1['target']
db_x_1 = database_1.drop(['target'], 1)
db_x_1.head()

# use this function to find r2 of redundant features 
# dropping a feature and seeing if the other features can predict it
def calculate_r_2_for_feature_tree(data,feature):
    new_data = data.drop(feature, axis=1)

    X_train,     X_test,      y_train,     y_test = train_test_split(
        new_data,data[feature],test_size=0.25
    )

    regressor = DecisionTreeRegressor()
    regressor.fit(X_train,y_train)

    score = regressor.score(X_test,y_test)
    return score

#use this function to take the mean of the scores after 100 runs
def mean_r2_for_feature_tree(data, feature):
    scores = []
    for _ in range(2):
        scores.append(calculate_r_2_for_feature_tree(data, feature))
        
    scores = np.array(scores)
    return scores.mean()

# use this function to get the mean of scores of multiple columns 
def mean_column_range_tree(l, h, data):  
    r2_tree= []
    for i in tqdm(data.columns[l:h]):
        if mean_r2_for_feature_tree(data, i) > 0:
            r2_tree.append(i)
    return r2_tree

# mean_column_range_tree(1,1001, db_x_1)

top_20 = ['feat_257', 'feat_269', 'feat_308', 'feat_315', 'feat_336', 'feat_341', 
                   'feat_395', 'feat_504', 'feat_526', 'feat_639', 'feat_681', 'feat_701', 
                   'feat_724', 'feat_736', 'feat_769', 'feat_808', 'feat_829', 'feat_867',
                   'feat_920', 'feat_956',]

def skb_top_feats(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=.2, 
                                                    random_state=42)
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    skb_list = []
    skb = SelectKBest(k=20)
    skb.fit(X_train, y_train)
    
    skb_feats = x.columns[skb.get_support()]
    
    skb_list.append(skb_feats)
    
    return skb_list

db_skb_feats = skb_top_feats(db_x_1, db_y_1)
db_skb_feats

db_skb_feats = ['feat_003', 'feat_257', 'feat_269', 'feat_308', 'feat_315', 'feat_336',
        'feat_341', 'feat_395', 'feat_504', 'feat_557', 'feat_681', 'feat_701',
        'feat_724', 'feat_736', 'feat_769', 'feat_783', 'feat_808', 'feat_829',
        'feat_867', 'feat_920']
for i in db_skb_feats:
    if i in top_20:
        print(i)

corr_df = db_x_1.corr().abs()
for i in corr_df.columns:
    corr_df.loc[i,i] = 0
corr_list = corr_df.max().sort_values(ascending=False)[:20]

corr_list = pd.DataFrame(corr_list)
corr_list.sort_index(ascending=True)

corr_list.sort_index(ascending=True)



