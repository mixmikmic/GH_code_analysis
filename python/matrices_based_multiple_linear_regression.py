import os  
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf

path = os.getcwd() + '\ex1data2.txt'  
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])  
data2.head()
num_samples = data2['Size'].count()
print ('num of samples: ',num_samples)

data2.describe()

fig = plt.figure()
ax=Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
ax.scatter(data2['Size'],data2['Bedrooms'],data2['Price'],c='blue',marker='o',alpha=0.5)
ax.set_xlabel('size')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
plt.show()

mean = data2.mean()
std = data2.std()
data_norm = (data2 - mean) / std  
data_norm.head()

data_norm.describe()

feature_names = ["Size", "Bedrooms"]
#data_x = data2.loc[:,columns]
data_x=data_norm[feature_names]
data_y = data_norm["Price"]
#print ('input_shape: ', data_x.shape)
#print ('output_shape: ', data_y.shape)
#data_x1 = data_norm["Size"]
#data_x2 = data_norm["Bedrooms"]
#print ('single_input_shape: ', data_x1.shape)
data_x = np.array(data_x,dtype='float32')
data_y = np.array(data_y,dtype='float32')
num_features=data_x.shape[1]
print ('num of features: ',num_features)

fig = plt.figure()
ax=Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_x[0:,0],data_x[0:,1],data_y,c='blue',marker='o',alpha=0.5)
ax.set_xlabel('size')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
plt.show()

with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [num_samples, 2], name="inputs")
    Y = tf.placeholder(tf.float32, [num_samples, 1],name="true_output")
    #X = tf.placeholder(tf.float32, [None, 2], name="inputs")
    #Y = tf.placeholder(tf.float32, [None, 1],name="true_output")

with tf.name_scope('parameters'):
    W = tf.Variable(tf.zeros([num_features,1]), name="weights")
    b = tf.Variable(tf.zeros([1]), name="bias")

with tf.name_scope('regression_model'):
  Y_predicted = tf.matmul(data_x,W) + b

with tf.name_scope('loss_function'):
    loss = tf.reduce_sum(tf.square(Y-Y_predicted,name = 'loss'))/(2*num_samples)
   # loss = tf.reduce_mean(tf.square(Y-Y_predicted,name = 'loss'))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# Add summary ops to collect data
# Add summary ops to collect data
W_hist = tf.summary.histogram("weights", W)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y_predicted", Y_predicted)
cost = tf.summary.scalar('loss',loss)
# merge all the summaries
merged_summaries = tf.summary.merge_all()

# Create a Saver object
saver = tf.train.Saver()

data_x_f = np.reshape(data_x,(num_samples,2))
data_y_f = np.reshape(data_y,(num_samples,1))
#final_data_x = np.array([[x[0], x[1]] for x in data_x]) 
#print('input_shape: ',data_x_f.shape)
#print('out_shape: ',data_y_f.shape)
cost_history = np.empty(shape=[1],dtype=float)
with tf.Session() as sess:
    # create a summary writer
    summary_writer = tf.summary.FileWriter('./matrices_mult_lin_reg_summary',sess.graph)
    # initialize the defined w and b variables
    sess.run(tf.global_variables_initializer())
    # Train the model
    for i in range(1000): # train the model for 100 iterations
        # for every iteration all the data is passed
        #for x,y in zip(data_x_f,data_y_f):
            #print (x.shape)
            #print(np.matrix(y).shape)
            # run the trining function to minimize the loss using 
            #defined optimizer
        _,loss_v,summary=sess.run([train_op,loss,merged_summaries],feed_dict={X:data_x_f,Y:data_y_f})
        cost_history=np.append(cost_history,loss_v)
            #summary_writer.add_summary(summary,i)
        # output the weight and bias value after every iteration
        if i%20==0:
            print ('loss is: ',loss_v)
            summary_writer.add_summary(summary,i)
    w_value,b_value = sess.run([W,b])
        #print (w_value,b_value)
        # print the loss function after every iteration
        #loss_value = sess.run(loss)
    saver.save(sess, '.\saved_model_matrices_mult\model_final_matrices_mult_lin_reg')
summary_writer.close()

len = cost_history.shape[0]
print (len)
iters = len
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iters), cost_history, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')

x_test = np.array(data_x[0:5])

y_test = np.array(data_y[0:5])
y_test_predicted=tf.matmul(x_test,w_value)+b_value
print ('predicted_value: ', y_test_predicted)
print ('true_value: ', y_test)
with tf.Session() as sess:
    sess.run(y_test_predicted)
    print (y_test_predicted.eval())

x_test = np.array(data_x)
y_test = np.array(data_y)
y_test_predicted=tf.matmul(x_test,w_value)+b_value
with tf.Session() as sess:
    sess.run(y_test_predicted)
    print (y_test_predicted.eval())

#plt.plot(x2_test,y_test,'o', x2_test,y_test_predicted,"-")
#plt.xlabel("num of bedrooms")
#plt.ylabel("price $")
#plt.legend(['data_X2:#bedrooms', 'Regression Model(a=w1*x1+w2*x2+b)'],bbox_to_anchor=(1, 1),loc=4)
#plt.show()

print('final weights: ', w_value)
print('final bias: ', b_value)


x1_surf,x2_surf= np.meshgrid(np.linspace(data_x[0:,0].min(),data_x[0:,0].max(),100),np.linspace(data_x[0:,1].min(),data_x[0:,1].max(),100))
#y_test_predicted_ = np.meshgrid(np.linspace(y_test_predicted.min(),y_test_predicted.max(),100))
Y_predicted_surf = x1_surf*w_value[0]+x2_surf*w_value[1]+b_value
print (Y_predicted_surf.shape)
#fittedY.reshape(x_surf.shape)
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax=Axes3D(fig)
sct = ax.scatter(data_x[0:,0],data_x[0:,1],data_y,c='blue', marker='o',alpha=0.5)
plt_surf = ax.plot_surface(x1_surf,x2_surf,Y_predicted_surf,color='red',alpha = 0.2)
ax.set_xlabel('size')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
plt.title('Multivariable Regression Model (a=w1*x1+w2*x2+b)')
plt.legend(['data_x:sie,data_y:bedrooms'],bbox_to_anchor=(1, 0.8),loc=4)
plt.show()

tf.reset_default_graph()  
with tf.Session() as sess:  
    imported_meta = tf.train.import_meta_graph(".\saved_model_matrices_mult\model_final_matrices_mult_lin_reg.meta")
    imported_meta.restore(sess, tf.train.latest_checkpoint('./saved_model_matrices_mult/'))
    w_final = sess.run(('parameters/weights:0'))
    b_final = sess.run(('parameters/bias:0'))
    print("wieights final: {}".format (w_final))
    print("bias final: {}".format (b_final))



