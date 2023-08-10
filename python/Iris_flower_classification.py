import pandas
import numpy
import keras
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = pandas.read_csv("data\iris_dataset.csv", header = None)
data.head()

plt.scatter(data.iloc[:,0:1][data[4]=='Iris-setosa'], data.iloc[:,1:2][data[4]=='Iris-setosa'], color ='green' , label='Iris-setosa')
plt.scatter(data.iloc[:,0:1][data[4]=='Iris-versicolor'], data.iloc[:,1:2][data[4]=='Iris-versicolor'], color ='red' , label='Iris-versicolor')
plt.scatter(data.iloc[:,0:1][data[4]=='Iris-virginica'], data.iloc[:,1:2][data[4]=='Iris-virginica'], color ='blue' , label='Iris-virginica')
plt.title("Iris plant dataset")
plt.xlabel("Sepal length in cm")
plt.ylabel("Sepal width in cm")
plt.legend()
plt.show()

plt.scatter(data.iloc[:, 2:3][data[4]=='Iris-setosa'], data.iloc[:, 3:4][data[4]=='Iris-setosa'], color ='green' , label='Iris-setosa')
plt.scatter(data.iloc[:, 2:3][data[4]=='Iris-versicolor'], data.iloc[:, 3:4][data[4]=='Iris-versicolor'], color ='red' , label='Iris-versicolor')
plt.scatter(data.iloc[:, 2:3][data[4]=='Iris-virginica'], data.iloc[:, 3:4][data[4]=='Iris-virginica'], color ='blue' , label='Iris-virginica')
plt.title("Iris plant dataset")
plt.xlabel("Petal length in cm")
plt.ylabel("Petal width in cm")
plt.legend()
plt.show()

dataset = data.values
X = dataset[: , 0:4]
y = dataset[: , 4]

encoder = LabelEncoder()
encoder.fit(y)
encoder_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoder_y)

def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3,activation='softmax'))
    # compile the model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

# fix random seed for reproducibility in evaluation
seed = 7
numpy.random.seed(seed)

kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline Prediction Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

results

