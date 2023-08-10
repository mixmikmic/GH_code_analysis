import numpy
from keras.models import Sequential
from keras.layers import Dense

seed = 7
numpy.random.seed(seed)

# Load the dataset. The 9th column is the target
dataset = numpy.loadtxt("../datasets/uci/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# Create the model
# Please refer to the above mentioned URL for explanations.
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

