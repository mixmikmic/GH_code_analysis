import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

# Read data from file
filename = '../../data/snow_fall.csv'
train_X = []
train_Y = []
with open(filename) as f:
    next(f)
    for line in f:
        temp = line.split(',')
        train_X.append(int(temp[0]))
        train_Y.append(float(temp[1]))
train_X = np.array(train_X)
train_Y = np.array(train_Y)

# train_X = np.asarray([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017])
# train_Y= np.asarray([40,39,41,29,32,30,33,15,10,11,20,24,10,15,18,12,17,15])

n_samples = train_X.shape[0]

print("Year Snowfall(inches)")
for y, s in zip(train_X, train_Y):
    print(y, s)

normalize = True

if normalize:
    mean_X = np.mean(train_X)
    mean_Y = np.mean(train_Y)
    std_X = np.std(train_X)
    std_Y = np.std(train_Y)
    train_X = (train_X - mean_X) / std_X
    train_Y = (train_Y - mean_Y) / std_Y

print("Samples: {}".format(train_X.shape[0]))
if normalize:
    plt.plot(train_X * std_X + mean_X, train_Y * std_Y + mean_Y, 'bo', label='Input data')
else:
    plt.plot(train_X, train_Y, 'bo', label='Input data')

plt.xlabel("year")
plt.ylabel("snowfall (inches)")
plt.legend()
plt.show()

# Parameters
learning_rate = 0.01
training_epochs = 100
display_steps=1

# Create placeholders for input X (height) and label Y (weight)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Create weight and bias, initialized to 0 (or rng.randn())
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Construct a linear model to predict Y
Y_predicted = tf.add(tf.multiply(X, w), b) # X * w + b 

# Mean squared error
loss = tf.reduce_sum(tf.pow(Y - Y_predicted, 2)) / (n_samples)  
# Using gradient descent optimizer to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Create a summary to monitor tensors
tf.summary.scalar("loss", loss)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

sess = tf.InteractiveSession()

# Initialize the necessary variables(i.e. assign their default value), in this case, w and b
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter('/tmp/linear_regression_snow', sess.graph)

# Train the model
for epoch in range(training_epochs):  # train the model for given number of epochs
    for (x, y) in zip(train_X, train_Y):
        _, summary = sess.run([optimizer, merged_summary_op], feed_dict={X: x, Y: y})
    summary_writer.add_summary(summary, epoch)
    # Display logs per epoch step
    if (epoch + 1) % display_steps == 0:
        l = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
        print('Epoch {}: loss={:.9f}, w={}, b={}'.format(epoch + 1, l, sess.run(w), sess.run(b)))

# Close the summary_writer when you're done using it
summary_writer.close()

print("Optimization Finished!")
training_loss = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
w_final, b_final = sess.run([w, b])
print('Training loss={}, w={}, b={}'.format(training_loss, w_final, b_final))

# plot the results
inputs = train_X
targets = train_Y
outputs = inputs * w_final + b_final

if normalize:
    inputs = inputs * std_X + mean_X
    targets = targets * std_Y + mean_Y
    outputs = outputs * std_Y + mean_Y

plt.plot(inputs, targets, 'bo', label='Real data')
plt.plot(inputs, outputs, 'r', label='Fitted line')
plt.legend()
plt.show()

# Inference phese
x = 2018

if normalize:
    x = (x - mean_X) / std_X

y = sess.run(Y_predicted, feed_dict={X: (x)})  # == x * w_final + b_final

if normalize:
    x = x * std_X + mean_X
    y = y * std_Y + mean_Y

print("Input = {}, Output = {:.2f} inches.".format(x, y))

