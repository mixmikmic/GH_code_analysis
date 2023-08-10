import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


mnist_data = np.load('mnist.npz') # import local npz 
X_train, y_train, X_test, y_test  = mnist_data['x_train'], mnist_data['y_train'], mnist_data['x_test'], mnist_data['y_test']
# normalize x
X_train = X_train.astype(float) / 255.
X_test = X_test.astype(float) / 255.
# we reserve the last 10000 training examples for validation
X_train, X_val = X_train[:-10000], X_train[-10000:]
y_train, y_val = y_train[:-10000], y_train[-10000:]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(X_train[1], cmap="Greys");

# feature number hight/width
n_feature1 = X_test.shape[1]
n_feature2 = X_test.shape[2]
n_feature = n_feature1 * n_feature2
m = X_train.shape[0]

tf.reset_default_graph()

# placeholder for data
X_input = tf.placeholder(dtype= tf.float32, shape=(None,n_feature1, n_feature2), name = 'X_input')
y_target = tf.placeholder(dtype=tf.int64, shape=(None,), name= 'y_target')

X_flatten = tf.reshape(X_input, shape=(-1, n_feature), name='X_input_reshape')

with tf.name_scope(name='layer_1') as scope:
    w1 = tf.Variable(tf.random_normal([n_feature, 384], stddev=0.01),  name= 'w1')
    b1 =  tf.Variable(tf.zeros([1,384]), name='b1')
    out1 = tf.nn.relu(tf.matmul(X_flatten, w1) + b1, name='out1')
#with tf.name_scope(name='layer_2') as scope:
#    w2 = tf.Variable(tf.random_normal([512, 128], stddev=0.01), name='w2')
 #   b2 =  tf.Variable(tf.zeros([1,128]), name='b2')
#    out2 = tf.nn.relu(tf.matmul(out1, w2) + b2, name='out2')
with tf.name_scope(name='output_layer') as scope:
    w3 = tf.Variable(tf.random_normal([384, 10], stddev=0.01), name ='w3')
    b3 =  tf.Variable(tf.zeros([1,10]), name='b3')
    logits = tf.nn.relu(tf.matmul(out1, w3) + b3, name='logits')
    y_predict = tf.nn.softmax(logits=logits)
    
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits= logits, labels= y_target))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)    

correct_prediction = tf.equal(tf.argmax(y_predict,1), y_target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
initializer = tf.global_variables_initializer()

epochs = 20
batch_size = 1000

loss_record = {'loss_train':[], 'loss_val':[]}

with tf.Session() as sess:
    sess.run(initializer)
    for epoch in range(epochs):
        for i in range(m // batch_size ):
            n_left = i * batch_size
            n_right = (i + 1)*batch_size
            X = X_train[n_left:n_right, :]
            y = y_train[n_left:n_right]
            loss_train, _, accuracy_train = sess.run([loss, optimizer, accuracy], feed_dict= {X_input:X, y_target:y})
            loss_val,accuracy_val =  sess.run([loss, accuracy], feed_dict= {X_input:X_val, y_target:y_val})
            loss_record['loss_train'].append(loss_train)
            loss_record['loss_val'].append(loss_val)
            '''if i % 2 == 0:
                print('epoch: ', epoch, '; batch: ', i)
                print('loss_train: ', loss_train, ';accuracy_train: ', accuracy_train, ';loss_val: ', loss_test, ';accuracy_val: ', accuracy_test)'''
    accuracy_test = sess.run([accuracy], feed_dict= {X_input:X_test, y_target:y_test}) 
    print('Test accuracy: ', accuracy_test)

fig, ax = plt.subplots()
ax.plot(loss_record['loss_train'], '--r', label='loss_train')
ax.plot(loss_record['loss_val'], 'b', label='loss_val')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper right', shadow=False)

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()



