import pandas as pd
import matplotlib
import numpy as np

data = pd.read_csv('AmesHousing.txt',sep='\t')
# remove recommended outliers
data = data.drop(data[data['Gr Liv Area'] > 4000].index)
print(len(data))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def transform_features(df):  
    return df

def select_features(df):
    return df[['Gr Liv Area','SalePrice']]

def train_and_test(df):
    k=10
    
    #split data into train and test
    train = df[:1460]
    test = df[1460:]
    
    #select numerical columns
    num_train = train.select_dtypes(include=['integer','float'])
    num_test = test.select_dtypes(include=['integer','float'])
    
    features = num_train.columns.drop('SalePrice')
    
    #create and fit model to train data
    lr = LinearRegression()
    lr.fit(num_train[features],num_train['SalePrice'])
    predictions = lr.predict(num_test[features])
    mse = mean_squared_error(test['SalePrice'],predictions)
    rmse = np.sqrt(np.absolute(mse))
    return rmse

transformed_data = transform_features(data)
filtered_data = select_features(transformed_data)
rmse = train_and_test(filtered_data)
print(rmse)

test_df = data[:]
# Let's keep track of these columns to drop later. We'll actually use Yr Sold in feature creation
nu_cols = ['Order','PID']
dl_cols = ['Mo Sold','Yr Sold','Sale Type','Sale Condition']

# Get number of missing values for each column.
num_missing = test_df.isnull().sum() 

# Remove text columns with any missing data
text_mv_counts = test_df.select_dtypes(include=['object']).isnull().sum()
text_mv_cols = text_mv_counts[text_mv_counts > 0].index
test_df = test_df.drop(text_mv_cols,axis=1)

# Remove numerical columns with >5% missing data
cutoff = test_df.shape[0]/20
print("5%: ", cutoff)
num_mv_counts = test_df.select_dtypes(include=['integer','float']).isnull().sum()
num_drop_cols = num_mv_counts[num_mv_counts > cutoff].index
test_df = test_df.drop(num_drop_cols,axis=1)
test_df.isnull().sum().value_counts()

# Find most common value for columns with missing data
num_mv_counts = test_df.select_dtypes(include=['integer','float']).isnull().sum()
num_mv_cols = num_mv_counts[num_mv_counts > 0].index
fill_values = test_df[num_mv_cols].mode().to_dict(orient='records')[0]
fill_values

# Fill in the missing values
test_df = test_df.fillna(fill_values)

test_df.isnull().sum().value_counts()

# Calculate values and check to see where they make sense
years_until_remod = data['Year Remod/Add'] - test_df['Year Built']
years_until_remod.value_counts()[[0,-1]]

# Create two features from 'Yr Sold'
years_until_sold = test_df['Yr Sold'] - test_df['Year Built']
print(years_until_sold[years_until_sold < 0])
years_since_remod = test_df['Yr Sold'] - test_df['Year Remod/Add']
print(years_since_remod[years_since_remod < 0])

test_df['years_until_sold'] = years_until_sold
test_df['years_since_remod'] = years_since_remod
test_df.drop(1702,axis=0)

# Drop year columns
test_df = test_df.drop(['Year Remod/Add','Year Built'],axis=1)

# Remove non-useful columns or columns that leak sale data
test_df = test_df.drop(nu_cols,axis=1)
test_df = test_df.drop(dl_cols,axis=1)

def transform_features(df):
    # Remove text columns with any missing data
    text_mv_counts = df.select_dtypes(include=['object']).isnull().sum()
    text_mv_cols = text_mv_counts[text_mv_counts > 0].index
    df = df.drop(text_mv_cols,axis=1)

    # Remove numerical columns with >5% missing data
    cutoff = df.shape[0]/20
    num_mv_counts = df.select_dtypes(include=['integer','float']).isnull().sum()
    num_drop_cols = num_mv_counts[num_mv_counts > cutoff].index
    df = df.drop(num_drop_cols,axis=1)
    
    # Find most common value for columns with missing data
    num_mv_counts = df.select_dtypes(include=['integer','float']).isnull().sum()
    num_mv_cols = num_mv_counts[num_mv_counts > 0].index
    fill_values = df[num_mv_cols].mode().to_dict(orient='records')[0]
    # Fill in the missing values
    df = df.fillna(fill_values)
    
    #Add Year-based features
    df['years_until_sold'] = df['Yr Sold'] - df['Year Built']
    df['years_since_remod'] = df['Yr Sold'] - df['Year Remod/Add']
    df.drop(1702,axis=0)
    
    # Drop year columns
    df = df.drop(['Year Remod/Add','Year Built'],axis=1)   
    # Remove non-useful columns or columns that leak sale data
    nu_cols = ['Order','PID']
    dl_cols = ['Mo Sold','Yr Sold','Sale Type','Sale Condition']
    df = df.drop(nu_cols,axis=1)
    df = df.drop(dl_cols,axis=1)
    return df
    
transformed_data = transform_features(data)
filtered_data = select_features(transformed_data)
rmse = train_and_test(filtered_data)
print(rmse)

get_ipython().magic('matplotlib inline')
import seaborn as sns
num_df = test_df.select_dtypes(include=['float','integer'])
sns.heatmap(num_df.corr().abs())

# Drop features that are highly correlated with other features
test_df = test_df.drop(['Garage Cars', 'Total Bsmt SF', 'TotRms AbvGrd'],axis=1)
num_df = num_df.drop(['Garage Cars', 'Total Bsmt SF', 'TotRms AbvGrd'],axis=1)

correlations = num_df.corr()['SalePrice'].abs().sort_values()
correlations

# Drop numerical columns with "SalePrice" correlation < .4.
test_df = test_df.drop(correlations[correlations < .4].index,axis=1)

# Select categorical columns
nominal_cols = ["PID","MS SubClass","MS Zoning","Street","Alley","Land Contour","Lot Config","Neighborhood","Condition 1","Condition 2","Bldg Type","House Style","Roof Style","Roof Matl","Exterior 1st","Exterior 2nd","Mas Vnr Type","Foundation","Heating","Central Air","Garage Type","Misc Feature","Sale Type"]
ordinal_cols = ["Lot Shape","Utilities","Land Slope","Exter Qual","Exter Cond","Bsmt Qual","Bsmt Cond","Bsmt Exposure","BsmtFin Type 1","BsmtFin Type 2","Heating QC","Electrical","Kitchen Qual","Functional","Fireplace Qu","Garage Finish","Garage Qual","Garage Cond","Paved Drive","Pool QC","Fence","Sale Condition"]
ordinal_numeric_cols = ["Overall Qual","Overall Cond"]
cat_cols = nominal_cols + ordinal_cols

transform_cat_cols = []
for column in cat_cols:
    if column in test_df.columns:
        transform_cat_cols.append(column)
cat_stats = test_df[transform_cat_cols].describe()
unique_stats = cat_stats.loc['unique']

# Only select categorical columns with <= 10 unique values
test_df = test_df.drop(unique_stats[unique_stats > 10].index,axis=1)

# Get rid of any categorical columns where the most frequent value is more than 95% of the total
top_percent_of_total = cat_stats.loc['freq']/cat_stats.loc['count']
test_df = test_df.drop(top_percent_of_total[top_percent_of_total > .95].index,axis=1)

test_df.describe(include=['object'])

# Convert to category columns
text_cols = test_df.select_dtypes(include=['object'])
for col in text_cols:
    test_df[col] = test_df[col].astype('category')
test_df = pd.concat([test_df,pd.get_dummies(test_df.select_dtypes(include=['category']))],axis=1)

def select_features(df,correlation_threshold=0.4,uniqueness_threshold=10):
    # Drop features that are highly correlated with other features
    df = df.drop(['Garage Cars', 'Total Bsmt SF', 'TotRms AbvGrd'],axis=1)
    
    num_df = df.select_dtypes(include=['float','integer'])
    correlations = num_df.corr()['SalePrice'].abs().sort_values()
    # Drop numerical columns with "SalePrice" correlation < our threshold.
    df = df.drop(correlations[correlations < correlation_threshold].index,axis=1)
    
    # Select categorical columns.
    nominal_cols = ["PID","MS SubClass","MS Zoning","Street","Alley","Land Contour","Lot Config","Neighborhood","Condition 1","Condition 2","Bldg Type","House Style","Roof Style","Roof Matl","Exterior 1st","Exterior 2nd","Mas Vnr Type","Foundation","Heating","Central Air","Garage Type","Misc Feature","Sale Type"]
    ordinal_cols = ["Lot Shape","Utilities","Land Slope","Exter Qual","Exter Cond","Bsmt Qual","Bsmt Cond","Bsmt Exposure","BsmtFin Type 1","BsmtFin Type 2","Heating QC","Electrical","Kitchen Qual","Functional","Fireplace Qu","Garage Finish","Garage Qual","Garage Cond","Paved Drive","Pool QC","Fence","Sale Condition"]
    cat_cols = nominal_cols + ordinal_cols
    
    # Transform categorical columns.
    transform_cat_cols = []
    for column in cat_cols:
        if column in df.columns:
            transform_cat_cols.append(column)
    cat_stats = df[transform_cat_cols].describe()
    unique_stats = cat_stats.loc['unique']

    # Drop categorical columns with more unique values than our threshold
    df = df.drop(unique_stats[unique_stats > uniqueness_threshold].index,axis=1)

    # Get rid of any categorical columns where the most frequent value is more than 95% of the total
    top_percent_of_total = cat_stats.loc['freq']/cat_stats.loc['count']
    df = df.drop(top_percent_of_total[top_percent_of_total > .95].index,axis=1)
    
    # Convert to category columns and create dummy columns
    text_cols = df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col] = df[col].astype('category')
    df = pd.concat([df,pd.get_dummies(df.select_dtypes(include=['category']))],axis=1)
    
    
    return df

transformed_data = transform_features(data)
filtered_data = select_features(transformed_data)
rmse = train_and_test(filtered_data)
print(rmse)

from sklearn.model_selection import KFold

def train_and_test(df,k=0):
    numeric_df = df.select_dtypes(include=['integer','float'])
    features = numeric_df.columns.drop('SalePrice')
    lr = LinearRegression()
    
    if k == 0:
        #split data into train and test
        train = numeric_df[:1460]
        test = numeric_df[1460:]

        #fit model to train data and test
        lr.fit(train[features],train['SalePrice'])
        predictions = lr.predict(test[features])
        mse = mean_squared_error(test['SalePrice'],predictions)
        rmse = np.sqrt(np.absolute(mse))

        return rmse
    
    if k == 1:
        # Randomize and split dataset
        numeric_df = numeric_df.reindex(np.random.permutation(numeric_df.index))
        fold_one = numeric_df[:1460]
        fold_two = numeric_df[1460:]
        
        # Train on 1 test on 2
        lr.fit(fold_one[features],fold_one['SalePrice'])
        predictions = lr.predict(fold_two[features])
        rmse_one = np.sqrt(mean_squared_error(fold_two['SalePrice'],predictions))
        
        # Train on 2 test on 1
        lr.fit(fold_two[features],fold_two['SalePrice'])
        predictions = lr.predict(fold_one[features])
        rmse_two = np.sqrt(mean_squared_error(fold_one['SalePrice'],predictions))
        
        #Average
        return (rmse_one + rmse_two) / 2
    
    # For k > 1
    kf = KFold(n_splits=k, shuffle=True)
    rmse_list = []
    # Iterate over all splits gathering train/test errors
    for train_index, test_index, in kf.split(numeric_df):
        train = numeric_df.iloc[train_index]
        test = numeric_df.iloc[test_index]
        lr.fit(train[features],train['SalePrice'])
        predictions = lr.predict(test[features])
        rmse = np.sqrt(mean_squared_error(test['SalePrice'],predictions))
        rmse_list.append(rmse)        
    # Return average of errors
    return np.mean(rmse_list)
        

transformed_data = transform_features(data)
filtered_data = select_features(transformed_data)
rmse = train_and_test(filtered_data,10)
print(rmse)

for k in range(10,21,1):
    f_data = select_features(transformed_data,correlation_threshold = .4,uniqueness_threshold = k)
    print(k,train_and_test(f_data,k))



