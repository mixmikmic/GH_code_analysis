import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
get_ipython().magic('matplotlib inline')
print("Package Loaded")

xy = np.loadtxt("data1.txt", delimiter=',', unpack=True, dtype='float64')
train_X = xy[0:-1]
train_Y = xy[-1]

n_samples = train_X[0].size

print ""
print "Type of 'train_X' is %s" % type(train_X)
print "Shape of 'train_X' is", train_X.shape
print ("Type of 'train_Y' is ", type(train_Y))
print ("Shape of 'train_Y' is", train_Y.shape)
print ("n_samples' is", n_samples)

pos = train_Y == 1
neg = train_Y == 0

plt.figure(1)
plt.plot(train_X[0][pos], train_X[1][pos], 'r+', label='Admitted')
plt.plot(train_X[0][neg], train_X[1][neg], 'bx', label='Not admitted')
plt.axis('equal')
plt.legend(loc='lower right')

temp_X = np.insert(train_X, 0, 1, axis=0)
temp_Y = train_Y.reshape([n_samples, 1])
#W = np.random.random((1, 3))
#print temp_Y.reshape(n_samples, 1)

def h(X, w):
    return tf.matmul(w, X)

def hypothesis(X, w):
    return tf.sigmoid(h(X, w))

def costF(X, Y, w):
    #return tf.reduce_mean(-Y*tf.log(hypothesis(X, w)) - (1-Y)*tf.log(1-hypothesis(X, w)))
    return -(tf.matmul(tf.log(hypothesis(X, w)), Y) + tf.matmul(tf.log(1-hypothesis(X, w)), 1-Y)) / n_samples
    #return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(hypothesis(X, w), Y))

X = tf.placeholder(tf.float32, name="input")
Y = tf.placeholder(tf.float32, name="output")
W = tf.Variable(tf.zeros([1, 3]), "weight")

learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(costF(X, Y, W))

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)
#feed = {X:temp_X, Y:temp_Y}
#temp = tf.reshape(Y, [n_samples, 1])
#print sess.run(temp, feed)
#print sess.run(W), sess.run(costF(X, Y, W), feed_dict={X:temp_X, Y:temp_Y})
start = time.time()
for step in xrange(50001):
    feed = {X:temp_X, Y:temp_Y}
    sess.run(train, feed)
    if step % 10000 == 0:
        print step, sess.run(costF(X, Y, W), feed), sess.run(W)
print time.time() - start

x = np.array([np.min(temp_X[1,:]), np.max(temp_X[1,:])])
y = (-1./sess.run(W[0,2])*(sess.run(W[0,0]) + sess.run(W[0,1])*x))

plt.figure(1)
plt.plot(train_X[0][pos], train_X[1][pos], 'r+', label='Admitted')
plt.plot(train_X[0][neg], train_X[1][neg], 'bx', label='Not admitted')
plt.plot(x, y, 'r-', label='Decision Boundary')
plt.axis('equal')
plt.legend(loc='lower right')



