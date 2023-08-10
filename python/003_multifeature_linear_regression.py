import tensorflow as tf
import pandas as pd
import numpy as np

np.random.seed(7)
tf.set_random_seed(7)
init_data = pd.read_csv("kc_house_data.csv")
print("Col: {0}".format(list(init_data)))

print(init_data.info())

init_data = init_data.drop("id", axis=1)
init_data = init_data.drop("date", axis=1)

matrix_corr = init_data.corr()
print(matrix_corr["price"].sort_values(ascending=False))

init_data = init_data.drop("zipcode", axis=1)

from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
## attributes to look at
attr = ["price", "sqft_living", "grade", "sqft_above", "sqft_living15"]
scatter_matrix(init_data[attr], figsize=(20,8) );

def split_data(data, ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_data(init_data, 0.2)

mean_offset = train_set.mean()
stddev_offset = train_set.std()
mean_xOffset = (mean_offset.drop("price")).values
stddev_xOffset = (stddev_offset.drop("price")).values
mean_yOffset = np.array([mean_offset["price"].copy()])
stddev_yOffset = np.array([stddev_offset["price"].copy()])

mean_xOffset = mean_xOffset.reshape(1, mean_xOffset.shape[0])
stddev_xOffset = stddev_xOffset.reshape(1, stddev_xOffset.shape[0])
mean_yOffset = mean_yOffset.reshape(1, mean_yOffset.shape[0])
stddev_yOffset = stddev_yOffset.reshape(1, stddev_yOffset.shape[0])

data = (train_set.drop("price", axis=1)).values
data_labels = (train_set["price"].copy()).values
data_labels = data_labels.reshape([len(data_labels),1]) # forcing a [None, 1] shape

num_features = data.shape[1]
n_samples = data.shape[0]

X_init = tf.placeholder(tf.float32, [None, num_features])
Y_init = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([num_features,1]))
b = tf.Variable(tf.random_normal([1]))

##  Grab the mean and stddev values we took from the training set earlier
x_mean = tf.constant(mean_xOffset, dtype=tf.float32)
y_mean = tf.constant(mean_yOffset, dtype=tf.float32)

x_stddev = tf.constant(stddev_xOffset, dtype=tf.float32)
y_stddev = tf.constant(stddev_yOffset, dtype=tf.float32)

## Making the input have a mean of 0 and a stddev of 1
X = tf.div(tf.subtract(X_init, x_mean), x_stddev)
Y = tf.div(tf.subtract(Y_init, y_mean), y_stddev)

pred = tf.add(tf.matmul(X,W), b)

pow_val = tf.pow(tf.subtract(pred, Y),2)
cost = tf.reduce_mean(pow_val)

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
#optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

cost_values = []
num_epochs = 1000
for epoch in range(num_epochs): 
    _, c = sess.run([optimizer, cost], feed_dict={X_init:data, Y_init:data_labels})
    cost_values.append(c)

training = sess.run(cost, feed_dict={X_init:data, Y_init:data_labels})
print("Final cost: {0} final weights: {1} final biases: {2}".format(training, sess.run(W), sess.run(b)) )

import matplotlib.pyplot as plt
plt.figure(1)
plt.title("Cost values")
plt.plot(cost_values);

## Defining R^2
ss_e = tf.reduce_sum(tf.pow(tf.subtract(Y, pred), 2))
ss_t = tf.reduce_sum(tf.pow(tf.subtract(Y, 0), 2))
r2 = tf.subtract(1.0, tf.div(ss_e, ss_t))

## Defining Adjusted R^2
adjusted_r2 = tf.subtract(1.0, tf.div(tf.div(ss_e, (n_samples - 1.0)), tf.div(ss_t, (n_samples - num_features - 1)) ) )

pred_data = sess.run(pred, feed_dict={X_init:data, Y_init:data_labels})
std_y_data = sess.run(Y, feed_dict={Y_init:data_labels})
## computing rmse
rmse = np.sqrt(np.mean(np.power(np.subtract(pred_data, std_y_data), 2)))
print("rmse of pred_data and std_y_data is: {0}".format(rmse))
print("R^2 value: {0}".format(sess.run(r2,feed_dict={X_init:data, Y_init:data_labels})) )
print("Adjusted R^2 value: {0}".format(sess.run(adjusted_r2, feed_dict={X_init:data, Y_init:data_labels})))

rmse_train = np.sqrt(np.mean(np.power(np.subtract(pred_data, std_y_data), 2)))
print("RMSE for Training data is: {0}".format(rmse_train))

## run test set through model
test_data = (test_set.drop("price", axis=1)).values
test_data_labels = (test_set["price"].copy()).values
test_data_labels = test_data_labels.reshape([len(test_data_labels), 1])
test_pred = sess.run(pred, feed_dict={X_init:test_data, Y_init:test_data_labels})
test_data_labels = sess.run(Y, feed_dict={Y_init:test_data_labels})
rmse_test = np.sqrt(np.mean(np.power(np.subtract(test_pred, test_data_labels), 2)))
print("RMSE for Test data is: {0}".format(rmse_test))
print("RMSE difference: {0}".format(rmse_test - rmse_train))
sess.close()



