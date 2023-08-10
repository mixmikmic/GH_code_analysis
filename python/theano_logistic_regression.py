import numpy as np
import theano
import theano.tensor as T
rng = np.random

#We generate a random dataset
N     = 400  # number of samples
feats = 80  # dimension of X

reg = 0.01 
lrate = 0.1
train_steps = 10000

D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

#We declare the variables of our model
w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0., name='b')

print("Initial model:")
print(w.get_value())
print(b.get_value())

#We declare the entry points and construct the graph
x = T.dmatrix('x')
y = T.dvector('y')

prob = 1 / (1 + T.exp(-T.dot(x, w) - b)) #logistic function 
xent = -y * T.log(prob) - (1-y) * T.log(1-prob)  #cross-entropy as error function
cost = xent.mean() + reg * (w ** 2).sum() #cost function with added regularization

#gradients we'll need for updates
gw = T.grad(cost, w) 
gb = T.grad(cost, b)

pred = prob > 0.5 #final classification

train = theano.function(
            inputs =[x,y],
            outputs=[cost],
            updates =((w, w - lrate * gw), (b, b - lrate * gb))
)
predict = theano.function(inputs=[x], outputs=pred)

#Training
for i in range(train_steps):
    err = train(D[0], D[1])
    
#Verification
score = np.mean(predict(D[0]) == D[1])
print(score)

#We take a look at our weights and bias if we are interested in them
print(w.get_value())
print(b.get_value())



