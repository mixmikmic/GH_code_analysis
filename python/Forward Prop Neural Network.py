import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

X = X / np.amax(X, axis=0)
y = y / 100

# Inspecting X
X

# Inspecting y
y

class NeuralNetwork(object):
    def __init__(self):
        # Hyperparemeters
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1
        
        # Weights for input -> hidden
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        # Weights for hidden -> output
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def forward(self, X):
        # Dot product of X (input) and first set of 3 x 2 weights
        self.z = np.dot(X, self.W1)
        # Applying the activation function
        self.z2 = self.sigmoid(self.z)
        # Dot product of the hidden layer and second set of 3 x 1 weights
        self.z3 = np.dot(self.z2, self.W2)
        # Final activation function
        o = self.sigmoid(self.z3)
        return o

brain = NeuralNetwork()

o = brain.forward(X)

print('Predicted result ', o)
print('Actual output ', y)

