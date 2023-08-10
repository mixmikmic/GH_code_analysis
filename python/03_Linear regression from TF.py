import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
print("Package Loaded")

np.random.seed(1)
def f(x, a, b):
    n = train_X.size
    vals = np.zeros((1, n))
    for i in range(0, n):
        ax = np.multiply(a, x.item(i))
        val = np.add(ax, b)
        vals[0, i] = val
    return vals

Wref = 0.7
bref = -1.
n = 20
noise_var = 0.001
train_X = np.random.random((1, n))
ref_Y = f(train_X, Wref, bref)
train_Y = ref_Y + np.sqrt(noise_var)*np.random.randn(1, n)
n_samples = train_X.size

print ""
print "Type of 'train_X' is %s" % type(train_X)
print "Shape of 'train_X' is", train_X.shape
print ("Type of 'train_Y' is ", type(train_Y))
print ("Shape of 'train_Y' is", train_Y.shape)

plt.figure(1)
plt.plot(train_X[0, :], ref_Y[0, :], 'ro', label='Original data')
plt.plot(train_X[0, :], train_Y[0, :], 'bo', label='Training data')
plt.axis('equal')
plt.legend(loc='lower right')

temp_X = np.insert(train_X, 0, 1, axis=0)
temp_Y = train_Y

def h(X, w):
    return tf.matmul(w, X)

def costF(X, Y, w):
    return tf.matmul((h(X, w) - Y), (h(X, w) - Y), transpose_b=True) / (2*n_samples)

X = tf.placeholder(tf.float64, name="input")
Y = tf.placeholder(tf.float64, name="output")
W = tf.cast(tf.Variable(tf.random_normal([1, 2]), "weight"), tf.float64)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)#.minimize(cost)
#optimizer.minimize(costF(X, Y, W))
#train = optimizer.minimize(costF(X, Y, W))
train = optimizer.apply_gradients(optimizer.compute_gradients(costF(X, Y, W)))

init = tf.initialize_all_variables()

sess = tf.Session()
#with tf.Session() as sess:
sess.run(init)
# print sess.run(costF(X, Y, W), feed_dict={X:temp_X, Y:temp_Y})
for step in range(20001):
    feed = {X:temp_X, Y:temp_Y}
    sess.run(train, feed)
    # sess.run(train, feed_dict={X:temp_X, Y:temp_Y})
    if step % 1000 == 0:
        print step, sess.run(costF(X, Y, W), feed), sess.run(W)

plt.figure(1)
plt.plot(train_X[0, :], ref_Y[0, :], 'ro', label='Original data')
plt.plot(train_X[0, :], train_Y[0, :], 'bo', label='Training data')
plt.plot(train_X[0, :], sess.run(h(X, W), feed_dict = {X:temp_X})[0, :], 'k', label='Fitting Line')
plt.axis('equal')
plt.legend(loc='lower right')



