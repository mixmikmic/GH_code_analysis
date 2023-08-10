import random
class Perceptron:
    def __init__(self, learn_speed, num_weights):
        self.speed = learn_speed
        self.weights = []
        for x in range(0, num_weights):
            self.weights.append(random.random()*2-1)
        #print("Self weights are ",self.weights)
        
    def feed_forward(self, inputs):
        sum = 0
        # multiply inputs by weights and sum them
        for x in range(0, len(self.weights)):
            sum += self.weights[x] * inputs[x]
        # return the 'activated' sum
            #print("\n Activate sum is ",self.activate(sum))
        return self.activate(sum)
    
    def activate(self, num):
        # turn a sum over 0 into 1, and below 0 into -1
        if num > 0:
            return 1
        return -1
    
    def train(self, inputs, desired_output):
        guess = self.feed_forward(inputs)
        error = desired_output - guess
        for x in range(0, len(self.weights)):
            self.weights[x] += error*inputs[x]*self.speed

class Trainer:
    def __init__(self):
        self.perceptron = Perceptron(0.01, 3)
    def f(self, x):
        return 0.5*x + 10 # line: f(x) = 0.5x + 10
    def train(self):
        for x in range(0, 1000000):
            x_coord = random.random()*500-250
            #print "x_coord is ",x_coord
            y_coord = random.random()*500-250
            #print "y_coord is ",y_coord            
            line_y = self.f(x_coord)
            #print "Line_y is ",line_y
            if y_coord > line_y: # above the line
                answer = 1
                self.perceptron.train([x_coord, y_coord,1], answer)
            else: # below the line
                answer = -1
                self.perceptron.train([x_coord, y_coord,1], answer)
        return self.perceptron # return our trained perceptron

trainer = Trainer()
p = trainer.train()
print "( -7 , 9 ) : " , p.feed_forward([-7,9,1])
print "( 3 , 1 ) : " ,p.feed_forward([3,1,1])

