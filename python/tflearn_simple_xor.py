from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0], [1], [1], [0]]

input_layer = input_data(shape=[None, 2]) #input layer of size 2
hidden_layer = fully_connected(input_layer , 2, activation='tanh') #hidden layer of size 2
output_layer = fully_connected(hidden_layer, 1, activation='tanh') #output layer of size 1

regression = regression(output_layer , optimizer='sgd', loss='binary_crossentropy', learning_rate=5)
model = DNN(regression)

model.fit(X, Y, n_epoch=5000, show_metric=True);

[i[0] > 0 for i in model.predict(X)]

print('Weights in layer1: ', model.get_weights(hidden_layer.W), ', Biases in layer1: ', model.get_weights(hidden_layer.b))
print('Weights in layer2: ', model.get_weights(output_layer.W), ', Biases in layer2: ', model.get_weights(output_layer.b))



