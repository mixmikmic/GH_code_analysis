import numpy as np
import pandas as pd
from fancyimpute import KNN

m_dum = pd.read_csv("dummified_df.csv")
m_dum.head()

col_idx_df = pd.DataFrame({'colname':m_dum.columns.values})
col_idx_df
# idx to drop: [0,1,2,4,5,6,7,8,9,14,15]

# drop columns that are irrevelant to modeling 
# and original categorical features which had beend dummified
# columns of interest: column index 11 'Rating' OR column index 64 'PreciseRating'
# dataframe w/ displayed rating 
idx_to_drop = [0,1,2,4,5,6,7,8,9,14,15]
museum_d = m_dum.drop(m_dum.columns[idx_to_drop + [64]], axis=1)
# dataframe w/ precise rating
museum_p = m_dum.drop(m_dum.columns[idx_to_drop + [11]], axis=1)

museum_d.describe() # descri_pol and descri_sub have NAs

museum_d.describe().to_csv('summary_before_imputation.csv')

museum_d.columns

from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# m_x = museum_d.drop(museum_d.columns[[2]], axis=1) # drop rating
# m_y = museum_d['Rating']
# m_x = df.drop(df.columns[[193]], axis=1) # drop PreciseRating
# m_y = df['PreciseRating']
def explore_model(df, col_name_to_pred, col_idx_to_drop):
    np.random.seed(0)
    m_x = df.drop(df.columns[[col_idx_to_drop]], axis=1) 
    m_y = df[col_name_to_pred]
    n_samples = m_x.shape[0]
    n_features = m_x.shape[1]
    print 'number of samples:', n_samples
    print 'number of features', n_features
    
    # subset training and testing
    x_train, x_test, y_train, y_test = train_test_split(m_x, m_y, test_size=0.2, random_state = 0)
    
    # linear regression
    ols = linear_model.LinearRegression()
    ols.fit(x_train, y_train)

    print 'multilinear score:', ols.score(x_train, y_train)
    predicted = ols.predict(x_test)
    print 'multilinear MSE:', np.mean((predicted - y_test)**2)

    # store coefficients
    coef_linear = pd.DataFrame({'feature':m_x.columns, 'coef': ols.coef_})
    coef_linear = coef_linear.sort_values(by = 'coef', axis=0, ascending = False)
    
    # random forest (regression)
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    print 'random forest (regression) score:', rf.score(x_train, y_train)
    predicted = rf.predict(x_test)
    print 'random forest (regression) MSE:', np.mean((predicted - y_test)**2)
    
    # random forest (classification
    # convert predictor from numeric into string
    y_train = [str(i) for i in y_train]
    y_test = [str(i) for i in y_test]        
    np.random.seed(0)
    rf_c = RandomForestClassifier()
    rf_c.fit(x_train, y_train)
    predicted = rf_c.predict(x_test)
    err = 1 - np.mean(predicted == y_test)
    print 'random forest (classification) score:', rf_c.score(x_train, y_train)
    predicted = rf_c.predict(x_test)
    print 'random forest (classification) error:', err
    
    # feature importance
    feature_imprtance = zip(m_x.columns, rf.feature_importances_)
    dtype = [('feature', 'S10'), ('importance', 'float')]
    feature_imprtance = np.array(feature_imprtance, dtype = dtype)
    feature_sort = np.sort(feature_imprtance, order='importance')[::-1]
    print 'RF regression'
    print 'top 10 important features:'
    print feature_sort[0:10]
    
    feature_imprtance_c = zip(m_x.columns, rf_c.feature_importances_)
    dtype = [('feature', 'S10'), ('importance', 'float')]
    feature_imprtance_c = np.array(feature_imprtance_c, dtype = dtype)
    feature_sort_c = np.sort(feature_imprtance_c, order='importance')[::-1]
    print 'RF Classification'
    print 'top 10 important features:'
    print feature_sort_c[0:10]
    
    return coef_linear, feature_sort, feature_sort_c 

def cv_rf_class(k_grid, df, col_name_to_pred, col_idx_to_drop):
    err_lst = []
    score_lst = []
    length = df.shape[0]
    for k_val in k_grid:
        X_filled_knn = KNN(k = k_val).complete(df)
        knn_imputed_md = pd.DataFrame(data = X_filled_knn,
                                      index= range(0,length),
                                      columns = df.columns)
        
        m_x = knn_imputed_md.drop(knn_imputed_md.columns[[col_idx_to_drop]], axis=1)        
        m_y = [str(i) for i in knn_imputed_md[col_name_to_pred]]
        x_train, x_test, y_train, y_test = train_test_split(m_x, m_y, test_size=0.2, random_state = 0)        
        np.random.seed(0)
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        predicted = rf.predict(x_test)
        score_lst.append(rf.score(x_train, y_train))
        err_lst.append(1 - np.mean(predicted == y_test))
    result_df = pd.DataFrame({'k':k_grid, 'score':score_lst, 'error':err_lst})  
    result_df = result_df.sort_values(by = 'error', axis=0)
    return result_df

def cv_rf_regres(k_grid, df, col_name_to_pred, col_idx_to_drop):
    mse_lst = []
    score_lst = []
    length = df.shape[0]
    for k_val in k_grid:
        X_filled_knn = KNN(k = k_val).complete(df)
        knn_imputed_md = pd.DataFrame(data = X_filled_knn,
                                      index= range(0,length),
                                      columns = df.columns)
        m_x = knn_imputed_md.drop(knn_imputed_md.columns[[col_idx_to_drop]], axis=1)
        m_y = knn_imputed_md[col_name_to_pred]
        np.random.seed(0)
        rf = RandomForestRegressor()            
        x_train, x_test, y_train, y_test = train_test_split(m_x, m_y, test_size=0.2, random_state = 0)
        rf.fit(x_train, y_train)
        predicted = rf.predict(x_test)
        score_lst.append(rf.score(x_train, y_train))
        mse_lst.append(np.mean((predicted - y_test)**2))
    result_df = pd.DataFrame({'k':k_grid, 'score':score_lst, 'mse':mse_lst})  
    result_df = result_df.sort_values(by = 'mse', axis=0)
    return result_df

# imput NA with mean value
mean_impute_d = museum_d.copy(deep = True)
mean_impute_p = museum_p.copy(deep = True)

mean_impute_d['descri_pol'] = museum_d['descri_pol'].fillna(np.mean(museum_d['descri_pol']))
mean_impute_d['descri_sub'] = museum_d['descri_sub'].fillna(np.mean(museum_d['descri_sub']))

mean_impute_p['descri_pol'] = museum_p['descri_pol'].fillna(np.mean(museum_p['descri_pol']))
mean_impute_p['descri_sub'] = museum_p['descri_sub'].fillna(np.mean(museum_p['descri_sub']))

(museum_p.columns == 'PreciseRating').tolist().index(True)

(museum_d.columns == 'Rating').tolist().index(True)

explore_model(mean_impute_d,'Rating',[2, 193])

col_name_to_pred = 'Rating'
col_idx_to_drop = [2, 193]
length = museum_d.shape[0]
# KNN imputation
X_filled_knn = KNN(k = 40).complete(museum_d)
knn_imputed_md = pd.DataFrame(data = X_filled_knn,
                              index= range(0,length),
                              columns = museum_d.columns)

# random forest (regression)
m_x = knn_imputed_md.drop(knn_imputed_md.columns[[col_idx_to_drop]], axis=1)
m_y = knn_imputed_md[col_name_to_pred]
np.random.seed(0)
rf = RandomForestRegressor()            
x_train, x_test, y_train, y_test = train_test_split(m_x, m_y, test_size=0.2, random_state = 0)
rf.fit(x_train, y_train)
predicted = rf.predict(x_test)
print 'rf regression score:', rf.score(x_train, y_train)
print 'rf regression MSE:', np.mean((predicted - y_test)**2)

# random forest (classification)
m_y = [str(i) for i in knn_imputed_md[col_name_to_pred]]
x_train, x_test, y_train, y_test = train_test_split(m_x, m_y, test_size=0.2, random_state = 0)        
np.random.seed(0)
rf_c = RandomForestClassifier()
rf_c.fit(x_train, y_train)
predicted = rf_c.predict(x_test)
print 'rf classification score:', rf_c.score(x_train, y_train)
print 'rf classification error:', 1 - np.mean(predicted == y_test)

feature_imprtance = zip(m_x.columns, rf.feature_importances_)
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_imprtance = np.array(feature_imprtance, dtype = dtype)
feature_sort = np.sort(feature_imprtance, order='importance')[::-1]
print 'RF regression'
print 'top 10 important features:'
print feature_sort[0:10]

feature_imprtance_c = zip(m_x.columns, rf_c.feature_importances_)
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_imprtance_c = np.array(feature_imprtance_c, dtype = dtype)
feature_sort_c = np.sort(feature_imprtance_c, order='importance')[::-1]
print 'RF Classification'
print 'top 10 important features:'
print feature_sort_c[0:10]

pd.DataFrame({'colname':museum_p.columns.values})



