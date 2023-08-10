import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
training_data_df = pd.read_csv('sales_data_training.csv',dtype='float')
print(training_data_df)

# Pull out columns for X (data to train with) and Y (value to predict)


# We pass axis=1 parameter that tells it we want to drop a column and not a row.
#  Finally, we'll call .values to get back the result as an array.

X_training = training_data_df.drop('total_earnings',axis=1).values # Dropping total_earning column from the Data Frame

# Here X_training has all the column except 'total_earnings' column. Since its been dropped

print('X-TRAINING:')
print(X_training)


# Selecting only 'total_earning' column from the dataframe
Y_training = training_data_df[['total_earnings']].values
print('\n\n\nY-TRAINING:')
print(Y_training)

# Load testing data set from CSV file
test_data_df = pd.read_csv('sales_data_test.csv',dtype='float')

# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = test_data_df.drop('total_earnings',axis=1).values

print('X-TESTING:')
print(X_testing)

Y_testing = test_data_df[['total_earnings']].values
print('\n\n\nY-TESTING:')
print(Y_testing)

# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.

X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

print('X_scaled_training')
print(X_scaled_training) # Contains all columns except 'total_earnings' thats been scaled between 0 to 1.
print('\n\n\nY_scaled_training')
print(Y_scaled_training) # Conatins 'total_earning' columns thats been scaled between 0 to 1.

# The shape attribute for numpy arrays returns the dimensions of the array.

print('Dimensions of X_scaled_testing:',X_scaled_testing.shape)
print(X_scaled_testing.shape[0],'rows','',X_scaled_testing.shape[1],'columns')

print('\n\nDimensions of Y_scaled_testing:',Y_scaled_testing.shape)
print(Y_scaled_testing.shape[0],'rows','',Y_scaled_testing.shape[1],'columns')
print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))



