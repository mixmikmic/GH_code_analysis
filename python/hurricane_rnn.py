import pandas as pd
import numpy as np

data = pd.read_csv('2000-2009-data.csv', names=['Year', 'Month', 'Day', 'Hour', 'HurricaneNum', 'Name', 'Lat', 'Long', 'WindSpeed', 'Pressure'])

# Create a unique key for all of the hurricane
data['unique-key'] = data['Name'] + '-' + data['Year'].map(str) + '-' + data['HurricaneNum'].map(str)

# Delete the columns of information that we are not using so far
data.drop(['Name', 'HurricaneNum', 'Year'], axis = 1, inplace = True)

# Preview the first 5 rows of data
data.head(10)

# Remove hurricanes where pressure = 0
data = data[data['Pressure'] != 0]
data.head()

# Since our keys are strings, we enumerate them to access them as integers 
keys = list(enumerate(pd.unique(data['unique-key'])))

y = np.zeros((170))
for x in range(0,170):
    y[x] = len(pd.DataFrame(data[data['unique-key'] == keys[x][1]], columns = data.keys()).reset_index(drop = True))

# Now contains how many time instances of data (or rows) each hurricane contains 
hurricane_amount = pd.DataFrame(y)

# Total amount of hurricanes we have in our dataset 
print(len(pd.unique(data['unique-key'])))

# Provides statistical information about the DataFrame 
# From here we can see that the shortest hurricane only has 4 rows of information and the largest hurricane has 94 rows
hurricane_amount.describe()

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

# Print the hurricane key with the amount of data they contain
print ('Top 6 Hurricanes (in terms of data quantity)')
for x in hurricane_amount.nlargest(6, 0).index:
    print (keys[x][1], "-", hurricane_amount.loc[x][0])

# Print the distribution of data quantity for all 174 hurricanes
hurricane_amount.plot.hist(bins=20)
plt.show()

data.describe()

from geopy.distance import great_circle as vc
import math as Math

y = np.zeros((170))
data['distance'] = np.zeros((5033))
data['direction'] = np.zeros((5033))

# For all hurricanes
for x in range(0,170):
    t = pd.DataFrame(data[data['unique-key'] == keys[x][1]], columns = data.keys()).reset_index(drop = False)
    dst = 0
    prev = (0,0)
    
    # For all latitude and longitude points of hurricane, calculate the angle of travel and distance
    for p in zip(t['Lat'], t['Long']):
        if prev == (0,0):
            prev = p
            continue 
        # Stores the distance into the DataFrame
        data.set_value(t[(t['Lat'] == p[0]) & (t['Long'] == p[1])]['index'].values[0], 'distance', vc(prev,p).miles)
        
        dLon = p[1] - prev[1];
        y_x = Math.sin(dLon) * Math.cos(p[0]);
        x_x = Math.cos(p[1]) * Math.sin(p[0]) - Math.sin(p[1]) * Math.cos(p[0]) * Math.cos(dLon);
        brng = Math.degrees(Math.atan2(y_x, x_x)) 
        if (brng < 0):
            brng+= 360;
        
        # Stores the angle of travel into the DataFrame
        data.set_value(t[(t['Lat'] == p[0]) & (t['Long'] == p[1])]['index'].values[0], 'direction', brng)
        dst += vc(prev,p).miles
        prev = p
    y[x] = dst

# Now contains the distance between all given latitude and longitude points
hurricane_distance = pd.DataFrame(y)

# Columns have been added
data.head()

# Here we can see that the hurricane that traveled the least only traveled 65 miles, while the one that traveled the most traveled 8402 miles
hurricane_distance.describe()

# Print the hurricane key with the amount of data they contain
print ('Top 6 Hurricanes (in terms of distance traveled)')
for x in hurricane_distance.nlargest(6, 0).index:
    print (keys[x][1], "-", hurricane_distance.loc[x][0], "miles -", hurricane_amount.loc[x][0])

# Plotted the amount of hurricane distance traveled vs the amount of data they contain.
corr = plt.scatter(hurricane_distance[0], hurricane_amount[0])
plt.show()

# Graph the trajectories of the longest hurricanes (the ones that traveled the most)
for x in hurricane_amount.nlargest(3, 0).index:
    data[data['unique-key'] == keys[x][1]].plot(x='Lat', y='Long') 

# Graph the trajectories of the shortest hurricanes (the ones that traveled the least)
for x in hurricane_amount.nsmallest(3, 0).index:
    data[data['unique-key'] == keys[x][1]].plot(x='Lat', y='Long')

# Graph the trajectories of 3 random hurricanes 
for x in np.random.choice(170, 3):
     data[data['unique-key'] == keys[x][1]].plot(x='Lat', y='Long') 

# We are removing some outliers that contain too little or too much information to keep a more normal distribution.
cond = (hurricane_amount > 13) & (hurricane_amount < 60)
keys25 = []

for x in cond.index:
    if cond.loc[x][0]:
        keys25.append(keys[x][1])

word2keys = {}
for x in keys:
    word2keys[x[1]] = x[0]
    
df = data[data['unique-key'].isin(keys25)]
df.head()

# Total amount of hurricanes we have now 
print(len(pd.unique(df['unique-key'])))

# Description of our new dataset 
df.describe()

# Same thing we did before to view the data but now with the reduced dataset 
keys = list(enumerate(pd.unique(df['unique-key'])))

y = np.zeros((116))
for x in range(0,116):
    y[x] = len(pd.DataFrame(df[df['unique-key'] == keys[x][1]], columns = df.keys()).reset_index(drop = True))

hurricane_amount = pd.DataFrame(y)

# Now we can see that we have at least 14 rows of information per hurricane and at most 59.
hurricane_amount.describe()

print ('Top 6 Hurricanes (in terms of data quantity)')
for x in hurricane_amount.nlargest(6, 0).index:
    print (keys[x][1], "-", hurricane_amount.loc[x][0])
    
hurricane_amount.plot.hist(bins=20)
plt.show()

# Distribution of distance traveled in a 6 hour time interval for all hurricanes
dist = df[df['distance'] > 0]
dist = np.log(dist['distance'])
ser = pd.Series(dist)
ser.plot.kde()

# Distribution of angle traveled in a 6 hour time interval for all hurricanes
direc = df[df['direction'] > 0]
direc = np.log(direc['direction'])
ser = pd.Series(direc)
ser.plot.kde()

corr = plt.scatter(df['Lat'], df['Long'])
plt.grid()
plt.show()

# Assigning each point to a specific location in the grid. 
# For example, we will learn how a hurricane in quadrant 2 with move.
df['gridID'] = np.zeros(3775)

# These variable are hyperparameters
lat_interval = 20
long_interval = 40

df['gridID'] = (df['Lat'] - 9.500) / lat_interval + ( (df['Long'] + 107.700) * 6) / long_interval
df['gridID'] = round(df['gridID'])
    
df.head()

df.to_csv('checkpoint-dataframe.csv') # Save the dataframe to csv for checkpoint

# Load the preprocessed data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoint-dataframe.csv', index_col=0) 

df.head() # Check loaded data

df.drop(['Month', 'Day', 'Hour', 'Lat', 'Long', 'unique-key'], axis = 1, inplace = True)
temp_df = df

temp_df = temp_df[temp_df['distance'] > 0]
temp_df['distance'] = np.log(temp_df['distance'])

temp_df = temp_df[temp_df['direction'] > 0]
temp_df['direction'] = np.log(temp_df['direction'])

temp_df.head()

max(temp_df['gridID']) # Total grid spots

from sklearn.preprocessing import MinMaxScaler

# Normalize the values to predict them more easily in our model
scaler = MinMaxScaler(feature_range=(0, 1))
temp_df = pd.DataFrame(scaler.fit_transform(temp_df), columns=['WindSpeed', 'Pressure', 'Distance', 'Direction', 'gridID'])
temp_df.head()

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # Because index starts at 0
    result = []
    
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    result = np.array(result)
    row = len(result) * 0.85 # Amount of data to train on    
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import math, time

def build_model(layers):
    model = Sequential()

    for x in range(0,5):
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
        model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False)) 
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[2]))
    model.add(Activation("relu"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

window = 14 # Another hyperparameter
X_train, y_train, X_test, y_test = load_data(temp_df[::-1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

model = build_model([5, window, 1])

model.fit(X_train, y_train, batch_size=512, epochs=100, validation_split=0.1, verbose=0)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

pred = model.predict(X_train)

plt.figure(figsize=(15, 4), dpi=100)
plt.plot(pred, color='red', label='Prediction Value')
plt.plot(y_train, color='blue', label='Grid Locations')
plt.legend(loc='upper left')

plt.show()



