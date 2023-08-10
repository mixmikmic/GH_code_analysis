import os  
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf

tf.__version__

path = os.getcwd() + '\ex1data1.txt'  
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])  
data.head()

data.describe()

num_of_samples = data.shape[0]
print ('Num of samples: ', num_of_samples)

data_x,data_y = data['Population'],data['Profit']

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

plt.plot(data_x,data_y,'o')
plt.xlabel("population in 10000's")
plt.ylabel("profit in 10000 $")
plt.legend(['Data'],bbox_to_anchor=(1, 1),loc=4)
plt.show()

print (data_x[0:1].shape,data_y[0:1].shape)

print (data_x[0:5],data_y[0:5])

with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32 , name = "input")
    Y = tf.placeholder(tf.float32 , name = "output")

with tf.name_scope('parameters'):
    w = tf.Variable(0.0,name='weights')
    b = tf.Variable(0.0,name='bias')

with tf.name_scope('regression_model'):
    Y_predicted = X*w + b

with tf.name_scope('loss_function'):
    loss = tf.reduce_mean(tf.square(Y-Y_predicted,name = 'loss'))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# Add summary ops to collect data
W_hist = tf.summary.histogram("weights", w)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y_predicted", Y_predicted)

cost = tf.summary.scalar('loss',loss)

# merge all the summaries
merged_summaries = tf.summary.merge_all()

# Create a Saver object
saver = tf.train.Saver()

cost_history = np.empty(shape=[1],dtype=float)
with tf.Session() as sess:
    # create a summary writer
    summary_writer = tf.summary.FileWriter('./simple_lin_reg_summary',sess.graph)
    # initialize the defined w and b variables
    sess.run(tf.global_variables_initializer())
    
    # Train the model
    for i in range(300): # train the model for number of  iterations
        # for every iteration all the data is passed
        for x,y in zip(data_x,data_y):
            # run the training function to minimize the loss using 
            #defined optimizer
            _,loss_v,summary=sess.run([train_op,loss,merged_summaries],feed_dict={X:x,Y:y})
        cost_history=np.append(cost_history,loss_v)
            #summary_writer.add_summary(summary,i)
        # output the weight and bias value after every iteration
        if i%20==0:
            print ('loss is: ',loss_v)
            summary_writer.add_summary(summary,i)
    w_value,b_value = sess.run([w,b])
        #print (w_value,b_value)
        # print the loss function after every iteration
        #loss_value = sess.run(loss)
    # Save the final model
    saver.save(sess, '.\saved_model\model_final_lin_reg')
summary_writer.close()

len = cost_history.shape[0]
print (len)
iters = len
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iters), cost_history, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')

#test 
x_test = np.array(data_x[0:5])
y_test = np.array(data_y[0:5])
y_test_predicted=x_test*w_value+b_value
print ('predicted_value: ', y_test_predicted)
print ('true_value: ', y_test)

print (w_value,b_value)

x_test = np.array(data_x)
y_test = np.array(data_y)
y_test_predicted=x_test*w_value+b_value
#y_test_predicted

plt.plot(x_test,y_test,'o', x_test,y_test_predicted,"-")
plt.xlabel("population in 10000's")
plt.ylabel("profit in 10000 $")
plt.legend(['Data', 'Linear Regression Model(a=w*x+b)'],bbox_to_anchor=(1, 1),loc=4)
plt.show()

plt.scatter(x_test,y_test,marker="o")
plt.plot(x_test,y_test_predicted,"r-")
plt.xlabel("population in 10000's")
plt.ylabel("profit in 10000 $")
plt.legend(['Data', 'Linear Regression Model(a=w*x+b)'],bbox_to_anchor=(1, 1),loc=4)
plt.show()

tf.reset_default_graph()  
with tf.Session() as sess:  
    imported_meta = tf.train.import_meta_graph("./saved_model/model_final_lin_reg.meta")
    imported_meta.restore(sess, tf.train.latest_checkpoint('./saved_model/'))
    w_final = sess.run(('parameters/weights:0'))
    b_final = sess.run(('parameters/bias:0'))
    print("wieight final: {}".format (w_final))
    print("bias final: {}".format (b_final))

