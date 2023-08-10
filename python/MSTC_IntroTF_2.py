import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Number of data per-class
Ndata_class=100

group1 = np.random.multivariate_normal([-4, -4], 20*np.identity(2), size=Ndata_class)
group2 = np.random.multivariate_normal([4, 4], 20*np.identity(2), size=Ndata_class)

# Plot artificial data
plt.scatter(group1.T[0][:],group1.T[1][:])
plt.scatter(group2.T[0][:],group2.T[1][:],color='g')
plt.xlabel('x1',fontsize=18)
plt.ylabel('x2',fontsize=18)
plt.title('Artificial Data',fontsize=18)

x1 = np.arange(-10, 10, 0.1)

b=0.5
w1=-1.0


x2= w1 * x1 + b 

# Plot linear discrimination
plt.scatter(group1.T[0][:],group1.T[1][:])
plt.scatter(group2.T[0][:],group2.T[1][:],color='g')
plt.plot(x1,x2,color='r')
plt.xlabel('x1',fontsize=18)
plt.ylabel('x2',fontsize=18)
plt.title('Linear Discrimination',fontsize=18)


train_X = np.vstack((group1, group2))
pred= train_X.T[1] - w1 * train_X.T[0] - b 
plt.plot(pred)
plt.xlabel('x1',fontsize=18)
plt.ylabel('x2',fontsize=18)
plt.title('Classification: < 0 \'blue\' > 0 \'green\' ',fontsize=18)

x = np.arange(-10, 11)
plt.title('Sigmoid : logistic function',fontsize=18)
plt.xlabel('$x$')
plt.ylabel('$p(X)=1/(1+exp(-x))$',fontsize=16)
plt.plot(x, (1/(1+np.exp(-x))));


plt.plot((1/(1+np.exp(-pred))))
plt.ylim(-0.02, 1.02)

plt.title('Class-probabilities',fontsize=18)
plt.xlabel('$pred$',fontsize=16)
plt.ylabel('$p(X)=1/(1+exp(-pred))$',fontsize=16)

train_labels = np.array([0.0] * Ndata_class + [1.0] * Ndata_class)

# Cost function is cross-entropy
pred_prob=(1/(1+np.exp(-pred)))
cost = -sum((train_labels) * np.log(pred_prob + 1e-10) + (1-train_labels) * np.log(1-pred_prob + 1e-10))

print "cross-entropy: {}".format(cost)

# Inputs are now two-dimensional and come with labels "blue" or "green" (represented by 0 or 1)
X = tf.placeholder("float", shape=[None, 2])
labels = tf.placeholder("float", shape=[None])


# Set model weights and bias as before
#W = tf.Variable(tf.zeros([2, 1], "float"), name="weight")
#b = tf.Variable(tf.zeros([1], "float"), name="bias")

W=tf.constant([[1.0], [1.0]],name="weights")
b=tf.constant(-0.5,name="bias")


# Predictor is now the logistic function
#pred = tf.sigmoid(tf.to_double(tf.reduce_sum(tf.matmul(X, W), axis=[1]) + b))
pred = tf.sigmoid(tf.to_double(tf.reduce_sum(tf.matmul(X, W),1) + b))

# Cost function is cross-entropy
cost = -tf.reduce_sum(tf.to_double(labels) * tf.log(pred) + (1-tf.to_double(labels)) * tf.log(1-pred))


# Initializing the variables
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

# We stack our two groups of 2-dimensional points
train_X = np.vstack((group1, group2))

# labels to feed them
train_labels = np.array([0.0] * Ndata_class + [1.0] * Ndata_class)


with tf.Session() as sess:
    
    sess.run(init)
    
    pred, cost=sess.run([pred, cost], feed_dict={X: train_X, labels: train_labels})

        

plt.plot(pred)

print "cross-entropy: {}".format(cost)

# Inputs are now two-dimensional and come with labels "blue" or "green" (represented by 0 or 1)
X = tf.placeholder("float", shape=[None, 2])
labels = tf.placeholder("float", shape=[None])

# Set model weights and bias as before
W = tf.Variable(tf.zeros([2, 1], "float"), name="weight")
b = tf.Variable(tf.zeros([1], "float"), name="bias")

# Predictor is now the logistic function
#pred = tf.sigmoid(tf.to_double(tf.reduce_sum(tf.matmul(X, W), axis=[1]) + b))
pred = tf.sigmoid(tf.to_double(tf.reduce_sum(tf.matmul(X, W),1) + b))


# Cost function is cross-entropy
cost = -tf.reduce_sum(tf.to_double(labels) * tf.log(pred) + (1-tf.to_double(labels)) * tf.log(1-pred))

# Gradient descent
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

# We stack our two groups of 2-dimensional points
train_X = np.vstack((group1, group2))

# labels to feed them
train_labels = np.array([0.0] * Ndata_class + [1.0] * Ndata_class)


with tf.Session() as sess:
    sess.run(init)
    
    # We can Run the optimization algorithm several times
    for i in range(10):
        cost_out,W_out,b_out,_=sess.run([cost, W,b, optimizer], feed_dict={X: train_X, labels: train_labels})
        print("Epoch : %d Cost= %s "%(i,cost_out))
        print(W_out)
        print(b_out)
        
    

import matplotlib.cm as cm
import seaborn as sns

n_samples=200
batch_size=40

with tf.Session() as sess:
    # We stack our two groups of 2-dimensional points and label them 0 and 1 respectively
    train_X = np.vstack((group1, group2))

    # labels to feed them
    train_labels = np.array([0.0] * Ndata_class + [1.0] * Ndata_class)


    sess.run(init)

    # Run the optimization algorithm 1000 times
    for i in range(1000):
        # Select random minibatch
        indices = np.random.choice(n_samples, batch_size)
        X_batch, labels_batch = train_X[indices], train_labels[indices]
        sess.run(optimizer, feed_dict={X: X_batch, labels: labels_batch})

        
    # Plot the predictions: the values of p
    Xmin = np.min(train_X)-1
    Xmax = np.max(train_X)+1
    x = np.arange(Xmin, Xmax, 0.1)
    y = np.arange(Xmin, Xmax, 0.1)
    

    plt.scatter(group1.T[0][:],group1.T[1][:])
    plt.scatter(group2.T[0][:],group2.T[1][:],color='g')
    plt.xlim(Xmin, Xmax)
    plt.ylim(Xmin, Xmax)
    print('W = ', sess.run(W))
    print('b = ', sess.run(b))
    
    xx, yy = np.meshgrid(x, y)
    predictions = sess.run(pred, feed_dict={X: np.array((xx.ravel(), yy.ravel())).T})
    
    plt.title('Probability that model will label a given point "green"')
    plt.contour(x, y, predictions.reshape(len(x), len(y)), cmap=cm.BuGn, levels=np.arange(0.0, 1.1, 0.1))
    plt.colorbar()



