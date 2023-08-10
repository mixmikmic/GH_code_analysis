import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True) #one hot true does one hot encoding
tf.reset_default_graph()

epoch=10
display_epoch = 1
logs_path = 'mnist'
n_classes = 10  #Total 10 digits
batch_size = 100

with tf.name_scope('inputs') as inputs:
        x = tf.placeholder(tf.float32, shape=(None,784), name='x')
        y = tf.placeholder(tf.float32, shape=(None,10), name='y')

def weights_xavi(shape, xavier_params = (None, None)):
    with tf.variable_scope('Weights'):
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def neuralnet(la,shape=[None,None]):
    with tf.name_scope('neuralnet'):
        w=weights_xavi(shape,xavier_params=(shape[0],shape[1]))
        b=weights_xavi([shape[1],],xavier_params=(0,shape[1]))
        act=tf.add(tf.matmul(la,w),b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def network(data):
    l1=neuralnet(data,shape=[784,500])
    l1=tf.nn.relu(l1)                          #Layer1
    l2=neuralnet(l1,shape=[500,500])
    l2=tf.nn.relu(l2)                          #Layer2
    l3=neuralnet(l2,shape=[500,500])
    l3=tf.nn.relu(l3)                          #Layer3
    l4=neuralnet(l3,shape=[500,10])
    return l4    

with tf.name_scope('Model'): 
    pred = network(x)                      #we get a output of our network
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar("Loss",cost)         #we find the cost using soft cross entopy.
with tf.name_scope('ADAM'):
    optimizer = tf.train.AdamOptimizer().minimize(cost)  #We use Adam Optimizer
with tf.name_scope('Accuracy'):
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))   #Calculate Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()        
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())       #Initializes All the variables
    summary_writer = tf.summary.FileWriter(logs_path)  #WritesTheGraphForTensorboard
    summary_writer.add_graph(graph=sess.graph)        
    gstep=1
    for i in range(epoch):
            epoch_loss = 0             #calculate loss
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) #Selects input and label
                #The optimizer,Cost,Tensorboard Function are Run.
                _, c,summary = sess.run([optimizer, cost,merged_summary_op], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                summary_writer.add_summary(summary,gstep) #Writes Summary
                gstep+=1

            print('Epoch', i, 'completed out of',epoch,'loss:',epoch_loss)
    print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

print("Run the command line:\n"           "--> tensorboard --logdir='mnist' "           "\nThen open http://0.0.0.0:6006/ into your web browser")

