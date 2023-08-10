# Importing the libraries
import numpy as np
np.random.seed(123)  # for reproducible results
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Importing the dataset
df = pd.read_csv('credit_card.csv')

#INSPECTION
df.head() 

df.columns.values

missing_data = df.isnull().sum().sort_values(ascending = False)
sparsity = missing_data/missing_data.count()*100
missing_data = pd.concat([missing_data, round(sparsity).sort_values(ascending = False)], 
                          axis=1, keys=['Missing Total', 'Sparsity %'])
missing_data.head(5)

defaulted = df['defaulted'].value_counts()
print('\n Number of customers defaulted: {} out of {}'.format(defaulted[1], len(df)))

df[['SEX', 'defaulted']].groupby(['SEX'], as_index=False).mean().sort_values(by='defaulted', ascending=False)

df[['EDUCATION', 'defaulted']].groupby(['EDUCATION'], as_index=False).mean().sort_values(by='defaulted', ascending=False)

age_groups = df[['AGE', 'defaulted']].groupby(['AGE'], as_index=False).mean().sort_values(by='defaulted', ascending=False)
age_groups.head(10)

age_groups.tail(10)

g = sns.FacetGrid(df, col='defaulted')
g.map(plt.hist, 'AGE', bins=20) 

grid = sns.FacetGrid(df, col='defaulted', row='SEX', size=2.2, aspect=1.6)
grid.map(plt.hist, 'AGE', alpha=.5, bins=20)
grid.add_legend()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X.astype('float64'))

from minisom import MiniSom
som = MiniSom(x = 20, y = 20, input_len = 24, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
#Returns the distance map (matrix) of the weights (Mean Inter-neurons Distances (MID))
#Each cell is the normalised (0 to 1) sum of the distances between a neuron (winning node) and its neighbours.
plt.figure(figsize=(12,9))
pcolor(som.distance_map().T) 
colorbar() #legend (1 indicates highest MID - outliers)
markers = ['o', 's'] #circles(customer defaulted payment), squares(customer not defaulted payment)
colors = ['r', 'g'] #rad (customer defaulted payment), green(customer not defaulted payment)

for i, x in enumerate(X): #i (loops through y[index], x (loops through customers denoted as X))
    w = som.winner(x)
    plot(w[0] + 0.5, #co-ordinate along x-axis
         w[1] + 0.5, #co-ordinate along y-axis
         markers[y[i]], #y[i] == 1 (customer defaulted payment), y[i] == 0 (customer not defaulted payment)
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 18,
         markeredgewidth = 3)

plt.tight_layout()
plt.show()

mappings = som.win_map(X) 

print ('\nThe number of high risk customers defaulting in this group: {}'.format(len(mappings[6,7])))

frauds = np.concatenate((mappings[(6,7)], mappings[(2,4)], mappings[(18,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)

len(frauds)

# Creating the matrix of features
customers = df.iloc[:, 1:].values #drop customerID column [1]

# Creating the dependent variable
is_fraud = np.zeros(len(df))
for i in range(len(df)):
    if df.iloc[i,0].astype('float32') in frauds[:,0].astype('float32'):
        is_fraud[i] = 1
pd.value_counts(pd.Series(is_fraud))

print('\nThe total number of customers who are most likely to be defaulting are: {}'.format(np.count_nonzero(is_fraud)))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

classifier.add(Dense(input_dim = 24, units = 2, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Callback
from keras.callbacks import History
histories = History()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(customers, is_fraud, test_size=0.3)

classifier.fit(X_train, y_train, batch_size = 1, epochs = 2, validation_data = (X_test,y_test), callbacks = [histories])

y_pred = classifier.predict(customers)
y_pred = np.concatenate((df.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()[::-1][:len(y_pred)]]# sort customer with the highest frauldent probabilty. 
y_pred[:10]

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()  
    classifier.add(Dense(input_dim = 24, units = 2, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))  
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 1, epochs = 5)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()*100
variance = accuracies.std()*100

print('The mean accuracy of the model is: %.2f%%, with a (+/- %.2f%%) variance' % (mean, variance))

