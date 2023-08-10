# packages used for machine learning
import tensorflow as tf

# packages used for processing: 
import matplotlib.pyplot as plt # for visualization
import numpy as np

# for operating system related stuff
import os
import sys # for memory usage of objects
from subprocess import check_output

# import pandas for reading the csv files
import pandas as pd

# to plot the images inline
get_ipython().run_line_magic('matplotlib', 'inline')

# set the random seed to 3 so that the output is repeatable
np.random.seed(3)

# Input data files are available in the "../Data/" directory.

def exec_command(cmd):
    '''
        function to execute a shell command and see it's 
        output in the python console
        @params
        cmd = the command to be executed along with the arguments
              ex: ['ls', '../input']
    '''
    print(check_output(cmd).decode("utf8"))

# check the structure of the project directory
exec_command(['ls', '../..'])

''' Set the constants for the script '''

# various paths of the files
data_path = "../../Data" # the data path

dataset = "MNIST"

data_files = {
    'train': os.path.join(data_path, dataset, "train.csv"),
    'test' : os.path.join(data_path, dataset, "test.csv")
}

base_model_path = '../../Models'

current_model_path = os.path.join(base_model_path, "IDEA_1")

model_path_name = os.path.join(current_model_path, "Model1_v3")

# constant values:
highest_pixel_value = 255
train_percentage = 95
num_classes = 10
no_of_epochs = 500
batch_size = 64
hidden_neurons = 512

raw_data = pd.read_csv(data_files['train'])

n_features = len(raw_data.columns) - 1
n_examples = len(raw_data.label)
print n_features, n_examples

raw_data.head(10)

labels = np.array(raw_data['label'])

labels.shape

# extract the data from the remaining raw_data
features = np.ndarray((n_features, n_examples), dtype=np.float32)

count = 0 # initialize from zero
for pixel in raw_data.columns[1:]:
    feature_slice = np.array(raw_data[pixel])
    features[count, :] = feature_slice
    count += 1 # increment count

features.shape

# normalize the pixel data by dividing the values by the highest_pixel_value
features = features / highest_pixel_value

plt.imshow((features[:, 9]).reshape((28, 28)))

# shuffle the data using a random permutation
perm = np.random.permutation(n_examples)
features = features[:, perm]
labels = labels[perm]

random_index = np.random.randint(n_examples)
random_image = features[:, random_index].reshape((28, 28))
# use plt to plot the image
plt.figure().suptitle("Label of the image: " + str(labels[random_index]))
plt.imshow(random_image)

# function to split the data into train - dev sets:
def split_train_dev(X, Y, train_percentage):
    '''
        function to split the given data into two small datasets (train - dev)
        @param
        X, Y => the data to be split
        (** Make sure the train dimension is the first one)
        train_percentage => the percentage which should be in the training set.
        (**this should be in 100% not decimal)
        @return => train_X, train_Y, test_X, test_Y
    '''
    m_examples = len(X)
    assert train_percentage < 100, "Train percentage cannot be greater than 100! NOOB!"
    partition_point = int((m_examples * (float(train_percentage) / 100)) + 0.5) # 0.5 is added for rounding

    # construct the train_X, train_Y, test_X, test_Y sets:
    train_X = X[: partition_point]; train_Y = Y[: partition_point]
    test_X  = X[partition_point: ]; test_Y  = Y[partition_point: ]

    assert len(train_X) + len(test_X) == m_examples, "Something wrong in X splitting"
    assert len(train_Y) + len(test_Y) == m_examples, "Something wrong in Y splitting"

    # return the constructed sets

    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = split_train_dev(features.T, labels, train_percentage)

train_X.shape, train_Y.shape, test_X.shape, test_Y.shape

train_X = train_X.T; test_X = test_X.T
train_X.shape, test_X.shape

# check by plotting some image
random_index = np.random.randint(train_X.shape[-1])
random_image = train_X[:, random_index].reshape((28, 28))
# use plt to plot the image
plt.figure().suptitle("Label of the image: " + str(train_Y[random_index]))
plt.imshow(random_image)

# defining the Tensorflow graph for this task:
tf.reset_default_graph() # reset the graph here:

# define the placeholders:
tf_input_pixels = tf.placeholder(tf.float32, shape=(n_features, None))
tf_integer_labels = tf.placeholder(tf.int32, shape=(None,))

# image shaped pixels for the input_pixels:
tf_input_images = tf.reshape(tf.transpose(tf_input_pixels), shape=(-1, 28, 28, 1))
input_image_summary = tf.summary.image("input_image", tf_input_images)

# define the one hot encoded version fo the integer_labels
tf_one_hot_encoded_labels = tf.one_hot(tf_integer_labels, depth=num_classes, axis=0)
tf_one_hot_encoded_labels

# define the layer 0 biases:
lay_0_b = tf.get_variable("layer_0_biases", shape=(n_features, 1), initializer=tf.zeros_initializer())


# layer 1 weights 
lay_1_W = tf.get_variable("layer_1_weights", shape=(hidden_neurons, n_features), 
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
lay_1_b = tf.get_variable("layer_1_biases", shape=(hidden_neurons, 1), 
                            dtype=tf.float32, initializer=tf.zeros_initializer())

# layer 2 weights
lay_2_W = tf.get_variable("layer_2_weights", shape=(hidden_neurons, hidden_neurons), 
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
lay_2_b = tf.get_variable("layer_2_biases", shape=(hidden_neurons, 1), 
                            dtype=tf.float32, initializer=tf.zeros_initializer())

# layer 3 weights
lay_3_W = tf.get_variable("layer_3_weights", shape=(hidden_neurons, hidden_neurons), 
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
lay_3_b = tf.get_variable("layer_3_biases", shape=(hidden_neurons, 1), 
                            dtype=tf.float32, initializer=tf.zeros_initializer())

# layer 4 weights
lay_4_W = tf.get_variable("layer_4_weights", shape=(hidden_neurons, hidden_neurons), 
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
lay_4_b = tf.get_variable("layer_4_biases", shape=(hidden_neurons, 1), 
                            dtype=tf.float32, initializer=tf.zeros_initializer())

# layer 5 weights
lay_5_W = tf.get_variable("layer_5_weights", shape=(hidden_neurons, hidden_neurons), 
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
lay_5_b = tf.get_variable("layer_5_biases", shape=(hidden_neurons, 1), 
                            dtype=tf.float32, initializer=tf.zeros_initializer())

# layer 6 weights
lay_6_W = tf.get_variable("layer_6_weights", shape=(num_classes, hidden_neurons), 
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
lay_6_b = tf.get_variable("layer_6_biases", shape=(num_classes, 1), 
                            dtype=tf.float32, initializer=tf.zeros_initializer())

# forward computation:
z1 = tf.matmul(lay_1_W, tf_input_pixels) + lay_1_b
a1 = tf.nn.relu(z1)

z2 = tf.matmul(lay_2_W, a1) + lay_2_b
a2 = tf.nn.relu(z2)

z3 = tf.matmul(lay_3_W, a2) + lay_3_b
a3 = tf.nn.relu(z3)

z4 = tf.matmul(lay_4_W, a3) + lay_4_b
a4 = tf.nn.relu(z4) 

z5 = tf.matmul(lay_5_W, a4) + lay_5_b
a5 = tf.nn.relu(z5) 

z6 = tf.matmul(lay_6_W, a5) + lay_6_b
a6 = tf.nn.relu(z6)

# in the backward computations, there are no actiavtion functions
y_in_back = a6

a1_back = tf.nn.relu(tf.matmul(tf.transpose(lay_6_W), y_in_back) + lay_5_b)
a2_back = tf.nn.relu(tf.matmul(tf.transpose(lay_5_W), a1_back) + lay_4_b)
a3_back = tf.nn.relu(tf.matmul(tf.transpose(lay_4_W), a2_back) + lay_3_b)
a4_back = tf.nn.relu(tf.matmul(tf.transpose(lay_3_W), a3_back) + lay_2_b)
a5_back = tf.nn.relu(tf.matmul(tf.transpose(lay_2_W), a4_back) + lay_1_b)
a6_back = tf.nn.relu(tf.matmul(tf.transpose(lay_1_W), a5_back) + lay_0_b)

y_in_back

in_back_vector = tf.placeholder(tf.float32, shape=(num_classes, None))

# computations for obtaining predictions: 
pred1_back = tf.nn.relu(tf.matmul(tf.transpose(lay_6_W), in_back_vector) + lay_5_b)
pred2_back = tf.nn.relu(tf.matmul(tf.transpose(lay_5_W), pred1_back) + lay_4_b)
pred3_back = tf.nn.relu(tf.matmul(tf.transpose(lay_4_W), pred2_back) + lay_3_b)
pred4_back = tf.nn.relu(tf.matmul(tf.transpose(lay_3_W), pred3_back) + lay_2_b)
pred5_back = tf.nn.relu(tf.matmul(tf.transpose(lay_2_W), pred4_back) + lay_1_b)
pred6_back = tf.nn.relu(tf.matmul(tf.transpose(lay_1_W), pred5_back) + lay_0_b)

# generated digits:
generated_digits = pred6_back

x_out_back = a6_back
x_out_back, tf_input_pixels

x_out_back_image = tf.reshape(tf.transpose(x_out_back), shape=(-1, 28, 28, 1))
output_image_summary = tf.summary.image("output_image", x_out_back_image)

y_in_back

def normalize(x):
    '''
        function to range normalize the given input tensor
        @param 
        x => the input tensor to be range normalized
        @return => range normalized tensor
    '''
    x_max = tf.reduce_sum(x, axis=0)
    # return the range normalized prediction values:
    return (x / x_max)

# forward cost 
fwd_cost = tf.reduce_mean(tf.abs(normalize(y_in_back) - tf_one_hot_encoded_labels))
fwd_cost_summary = tf.summary.scalar("Forward_cost", fwd_cost)

# backward cost 
# The backward cost is the mean squared error function
bwd_cost = tf.reduce_mean(tf.abs(x_out_back - tf_input_pixels))
bwd_cost_summary = tf.summary.scalar("Backward_cost", bwd_cost)

cost = fwd_cost + bwd_cost
final_cost_summary = tf.summary.scalar("Final_cost", cost)

# define an optimizer for this task
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
init = tf.global_variables_initializer()
all_summaries = tf.summary.merge_all()

n_train_examples = train_X.shape[-1]

sess = tf.InteractiveSession()

tensorboard_writer = tf.summary.FileWriter(model_path_name, graph=sess.graph, filename_suffix=".bot")

sess.run(init)

# start training the network for num_iterations and using the batch_size
global_step = 0
for epoch in range(no_of_epochs):
    global_index = 0; costs = [] # start with empty list
    while(global_index < n_train_examples):
        start = global_index; end = start + batch_size
        train_X_minibatch = train_X[:, start: end]
        train_Y_minibatch = train_Y.astype(np.int32)[start: end]

        iteration = global_index / batch_size
        
        # run the computation:
        _, loss = sess.run((optimizer, cost), feed_dict={tf_input_pixels: train_X_minibatch, 
                                                         tf_integer_labels: train_Y_minibatch})

        # add the cost to the cost list
        costs.append(loss)

        if(iteration % 100 == 0):
            sums = sess.run(all_summaries, feed_dict={tf_input_pixels: train_X_minibatch, 
                                                         tf_integer_labels: train_Y_minibatch})
            
            print "Iteration: " + str(global_step) + " Cost: " + str(loss)

            tensorboard_writer.add_summary(sums, global_step = global_step)
        
        # increment the global index 
        global_index = global_index + batch_size
    
        global_step += 1
        
    # print the average epoch cost:
    print "Average epoch cost: " + str(sum(costs) / len(costs))
        

model_file_name = os.path.join(model_path_name, model_path_name.split("/")[-1])
model_file_name

saver = tf.train.Saver()

saver.save(sess, model_file_name, global_step=global_step)

saver.restore(sess, tf.train.latest_checkpoint(model_path_name))

# check by plotting some image
random_index = np.random.randint(test_X.shape[-1])
random_image = test_X[:, random_index].reshape((28, 28))
# use plt to plot the image
plt.figure().suptitle("Label of the image: " + str(test_Y[random_index]))
plt.imshow(random_image)

# generate the predictions for one random image from the test set.
predictions = np.squeeze(sess.run(y_in_back, feed_dict={tf_input_pixels: test_X[:, random_index].reshape((-1, 1))}))

plt.figure().suptitle("Predictions obtained from the network")
plt.plot(range(10), predictions);
print predictions
print "Predicted label: " + str(np.argmax(predictions))

tf_input_pixels, train_X.shape

preds = sess.run(y_in_back, feed_dict={tf_input_pixels: train_X})

correct = np.sum(np.argmax(preds, axis=0) == train_Y)
accuracy = (float(correct) / train_X.shape[-1]) * 100
print "Training accuracy: " + str(accuracy)

test_preds = sess.run(y_in_back, feed_dict={tf_input_pixels: test_X})
test_correct = np.sum(np.argmax(test_preds, axis=0) == test_Y)
test_accuracy = (float(test_correct) / test_X.shape[-1]) * 100
print "Testing accuracy:" + str(test_accuracy)

generator_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 20, 0]).reshape(-1, 1).astype(np.float32)
generator_array.dtype
generated_image = sess.run(generated_digits, feed_dict={in_back_vector: generator_array}).reshape((28, 28))
plt.imshow(generated_image)

total_frames = 50

all_digits = [] # start with an empty list
for walking_axis in range(num_classes):
    reps = np.zeros(shape=(num_classes, total_frames))
    for cnt in range(total_frames):
        reps[walking_axis, cnt] = cnt
    all_digits.append(reps)

all_digits = np.hstack(all_digits)

all_digits.shape

# obtain the images for these inputs:
images = sess.run(generated_digits, feed_dict={in_back_vector: all_digits}).T.reshape((-1, 28, 28))

images.shape

imagelist = images

import matplotlib.animation as animation
from IPython.display import HTML

fig = plt.figure() # make figure

# make axesimage object
# the vmin and vmax here are very important to get the color map correct
im = plt.imshow(imagelist[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=1);

# function to update figure
def updatefig(j):
    # set the data in the axesimage object
    im.set_array(imagelist[j])
    # return the artists set
    return [im]
# kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(images.shape[0]), 
                              interval=50, blit=True)

HTML(ani.to_html5_video())

