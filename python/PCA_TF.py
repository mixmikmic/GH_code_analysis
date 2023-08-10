import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

X = np.reshape(np.asarray((-1.64,-0.4,-1.84,-0.9,-0.84,0.1,2.16,0.1,2.16,1.1)),(5,2))
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
res = pca.fit_transform(X)
W = pca.transform(np.identity(2))
W #The transformation

W1 = np.reshape(W[:,0],(2,1))
#np.matmul(np.matmul(X, W1),np.transpose(W1))
X_rec = np.matmul(np.matmul(X, W1), np.transpose(W1))
np.sum((X - X_rec)**2)

tf.reset_default_graph()
X_ = tf.placeholder(name='X', shape=(None, 2), dtype='float32')
W = tf.get_variable('W', shape=(2,1))
X_rec_ = tf.matmul(tf.matmul(X_,W), tf.transpose(W))
loss_ = tf.reduce_sum((X_rec_ - X_)**2)

train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_) 
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(200):
        _, d, W_tf = sess.run([train_op, loss_, W], feed_dict={X_:X})
        if (i % 10 == 0):
            print(d)    
print("The transformation is {}".format(W_tf))

