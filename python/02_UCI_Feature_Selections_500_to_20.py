get_ipython().run_line_magic('run', '../__init__.py')
get_ipython().run_line_magic('matplotlib', 'inline')

target = pd.read_csv('../Datasets/madelon_train.labels', sep=' ', header=None)
train = pd.read_csv('../Datasets/madelon_train.data', sep=' ', header=None)
val = pd.read_csv('../Datasets/madelon_valid.data', sep=' ', header=None)

target.columns = ['target']
train = train.drop(train.columns[500], axis=1)

train= pd.concat([train, target], 1)

X = train.drop(['target'], axis=1)
y = train['target']

sample1 = train.sample(440)

Uci_y_1 = sample1['target']
Uci_X_1 = sample1.drop(['target'], axis=1)
Uci_X_1.shape

# use this function to find r2 of redundant features 
# dropping a feature and seeing if the other features can predict it
def calculate_r_2_for_feature(data,feature):
    new_data = data.drop(feature, axis=1)

    X_train,     X_test,      y_train,     y_test = train_test_split(
        new_data,data[feature],test_size=0.25
    )

    regressor = KNeighborsRegressor()
    regressor.fit(X_train,y_train)

    score = regressor.score(X_test,y_test)
    return score

#use this function to take the mean of the scores after 100 runs
def mean_r2_for_feature(data, feature):
    scores = []
    for _ in range(10):
        scores.append(calculate_r_2_for_feature(data, feature))
        
    scores = np.array(scores)
    return scores.mean()

# use this function to get the mean of scores of multiple columns 
def mean_column_range_Knn(data):
    r2_knn = []
    for i in tqdm(range(0,500)):
        if mean_r2_for_feature(data, i) > 0:
            r2_knn.append(i)
    return r2_knn

r2_knn = mean_column_range_Knn(Uci_X_1)

r2_knn

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

#use this function to take the mean of the scores after 5 runs
def mean_r2_for_feature_tree(data, feature):
    scores = []
    for _ in range(5):
        scores.append(calculate_r_2_for_feature_tree(data, feature))
        
    scores = np.array(scores)
    return scores.mean()

# use this function to get the mean of scores of multiple columns 
def mean_column_range_tree(data):
    r2_tree= []
    for i in tqdm(range(0,500)):
        if mean_r2_for_feature_tree(data, i) > 0:
            r2_tree.append(i)
    return r2_tree

r2_tree = mean_column_range_tree(Uci_X_1)

display(r2_tree)

top_20 = [28, 48, 64, 105, 128, 153, 241, 281, 318, 336, 338, 378, 433,
          442, 451, 453, 455, 472, 475, 493]

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
    skb.fit(X_train_scaled, y_train)
    
    skb_feats = np.where(skb.get_support())[0]
    
    skb_list.append(skb_feats)
    
    return skb_list

uci_skb_feats = skb_top_feats(Uci_X_1, Uci_y_1)

uci_skb_feats

uci_skb_feats = [48, 64, 105, 128, 137, 149, 199, 204, 241, 282, 329, 336, 338, 378,
        424, 442, 453, 472, 475, 493]

for i in uci_skb_feats:
    if i in top_20:
        print(i)

corr_df = X.corr().abs()
corr_df = corr_df > .5
corr_df = corr_df[corr_df].count()

corr_df = Uci_X_1.corr().abs()
for i in corr_df.columns:
    corr_df.loc[i,i] = 0
corr_list = corr_df.max().sort_values(ascending=False)[:20]

corr_list = pd.DataFrame(corr_list)
corr_list.sort_index(ascending=True)

train.to_pickle('./Datasets/train_clean.p')

