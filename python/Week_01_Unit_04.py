from numpy import exp, array, random, dot

class SingleNeuronNetwork():
    def __init__(self):
        # Set the seed for the random number generator
        # Ensures same random numbers are produced every time the program is run
        random.seed(42)

        # --- Model a single neuron: 3 input connections and 1 output connection ---
        # Assign random weights to a 3 x 1 matrix: Floating-point values in (-1, 1)
        self.weights = 2 * random.random((3, 1)) - 1

    # --- Define the Sigmoid function ---
    # Pass the weighted sum of inputs through this function to normalize between [0, 1]
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # --- Define derivative of the Sigmoid function ---
    # Evaluates confidence of existing learnt weights
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # --- Define the training procedure ---
    # Modufy weights by calculating error after every iteration
    def train(self, train_inputs, train_outputs, num_iterations):
        # We run the training for num_iteration times
        for iteration in range(num_iterations):
            # Feed-forward the training set through the single neuron neural network
            output = self.feed_forward(train_inputs)

            # Calculate the error in predicted output 
            # Difference between the desired output and the feed-forward output
            error = train_outputs - output

            # Multiply the error by the input and again by the gradient of Sigmoid curve
            # 1. Less confident weights are adjusted more
            # 2. Inputs, that are zero, do not cause changes to the weights
            adjustment = dot(train_inputs.T, error * 
                             self.__sigmoid_derivative(output))

            # Make adjustments to the weights
            self.weights += adjustment

    # --- Define feed-forward procedure ---
    def feed_forward(self, inputs):
        # Feed-forward inputs through the single-neuron neural network
        return self.__sigmoid(dot(inputs, self.weights))

# Intialise a single-neuron neural network.
neural_network = SingleNeuronNetwork()

print ("Neural network weights before training (random initialization): ")
print (neural_network.weights)

# The train data consists of 6 examples, each consisting of 3 inputs and 1 output
train_inputs = array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]])
train_outputs = array([[0, 1, 0, 1, 0, 1]]).T

# Train the neural network using a train inputs.
# Train the network for 10,000 steps while modifying weights to reduce error.
neural_network.train(train_inputs, train_outputs, 10000)

print ("Neural network weights after training: ")
print (neural_network.weights)

# Test the neural network with a new input
print ("Inferring predicting from the network for [1, 0, 0] -> ?: ")
print (neural_network.feed_forward(array([1, 0, 0])))

print ("Inferring predicting from the network for [0, 1, 1] -> ?: ")
print (neural_network.feed_forward(array([0, 1, 1])))



