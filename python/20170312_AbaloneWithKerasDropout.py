# Data preprocessing from Part 1
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
abalone_df = pd.read_csv('abalone.csv',names=['Sex','Length','Diameter','Height',
    'Whole Weight','Shucked Weight', 'Viscera Weight','Shell Weight', 'Rings'])
abalone_df['Male'] = (abalone_df['Sex']=='M').astype(int)
abalone_df['Female'] = (abalone_df['Sex']=='F').astype(int)
abalone_df['Infant'] = (abalone_df['Sex']=='I').astype(int)
abalone_df = abalone_df[abalone_df['Height']>0]
train, test = train_test_split(abalone_df, train_size=0.7)
x_train = train.drop(['Rings','Sex'], axis=1).values
y_train = pd.DataFrame(train['Rings']).values
x_test = test.drop(['Rings','Sex'], axis=1).values
y_test = pd.DataFrame(test['Rings']).values
# Constructing a list of models to test
hlayers = [[x,y] for x in range(5,31,5) for y in range(5,31,5)]
hlayers.extend([[1,10],[10,1],[2,2]])

# Iterate over the list of models, trying different dropout
begin = datetime.datetime.now()
results_dict = {}
for drop in [0.1, 0.2, 0.5, 0.75]:
    for layers in hlayers:
        abalone_model = Sequential([
            Dense(layers[0], input_dim=10),
            Dropout(drop),
            Dense(layers[1], activation='tanh'),
            Dropout(drop),
            Dense(1)])
        abalone_model.compile(optimizer='rmsprop',loss='mse',metrics=["mean_absolute_error"])
        results = abalone_model.fit(x_train, y_train, nb_epoch=50, verbose=0)
        score = abalone_model.evaluate(x_test, y_test)
        result_string = "[{},{}] drop={}".format(layers[0], layers[1], drop)
        results_dict[result_string] = score[1]
# Save the results in a DataFrame
results_df = pd.DataFrame.from_dict(results_dict, orient="index")
results_df.rename(columns={0 : "MAE"}, inplace=True)
seconds = (datetime.datetime.now() - begin).total_seconds()
sec_string = "Total elapsed seconds: {}".format(seconds)
print(sec_string)

# Print the results matrix
results_df.sort_values('MAE').head(15)

