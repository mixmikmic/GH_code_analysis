# Load pickled data and import dependencies
import pickle
import matplotlib.pyplot as plt
from skimage import exposure
get_ipython().magic('matplotlib inline')
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import scipy.ndimage
from matplotlib.gridspec import GridSpec


def get_model_dir():
    model_dir = '{}/{}'.format(os.getcwd(),'model_dir')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

filepath = '/Users/matthewzhou/Desktop/Zhou_Traffic_Signs'
training_file = filepath + '/train.p'
testing_file = filepath + '/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
train['features'] = rgb2gray(train['features'])
test['features'] = rgb2gray(test['features'])

mean = np.mean(train['features'])
print(mean)
mean = np.mean(test['features'])
print(mean)

mean = np.mean(train['features'])
train['features'] /= mean
mean = np.mean(test['features'])
test['features'] /= mean

train_dataset = train['features']
train_labels = train['labels']
test_dataset = test['features']
test_labels = test['labels']



train_dataset = np.array(train_dataset)
train_labels = np.array(train_labels)

inputs_per_class = np.bincount(train_labels)
max_inputs = np.max(inputs_per_class)

print('Preprocessing data...')
# Generate additional data for underrepresented classes
print('Generating additional data...')
angles = [-5, 5, -10, 10, -15, 15, -20, 20]

for i in range(len(inputs_per_class)):
    input_ratio = min(int(max_inputs / inputs_per_class[i]) - 1, len(angles) - 1)

    if input_ratio <= 1:
        continue

    new_features = []
    new_labels = []
    mask = np.where(train_labels == i)

    for j in range(input_ratio):
        for feature in train_dataset[mask]:
            new_features.append(scipy.ndimage.rotate(feature, angles[j], reshape=False))
            new_labels.append(i)

    train_dataset = np.append(train_dataset, new_features, axis=0)
    train_labels = np.append(train_labels, new_labels, axis=0)

train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(train_dataset, train_labels, test_size = 0.2)



num_channels = 1
image_size = 32
num_labels = 43

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



### To start off let's do a basic data summary.
import numpy as np
# TODO: number of training examples
n_train = len(train_dataset)

# TODO: number of testing examples
n_test = len(test_dataset)

# TODO: what's the shape of an image?
image_shape = train_dataset[0].shape

# TODO: how many classes are in the dataset
n_classes = len(np.unique(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.

def get_label_map(f_name, labels):
    mapper = np.genfromtxt(f_name, delimiter=',', usecols=(1,), dtype=str, skip_header=1)
    crapper = np.vectorize(lambda x : mapper[x])
    return crapper(labels)

def plot_class_frequencies(labels, settype):
    freqs = group(labels)
    plt.figure(figsize=(15,5))
    plt.title(settype + " Set Label Frequencies")
    plt.bar(freqs[:,0], freqs[:,1])
    plt.xlabel('ClassID')
    plt.ylabel('Frequency')
    ind = np.arange(0.5,43.5)
    plt.xticks(ind, get_label_map('signnames.csv', np.unique(labels)),  ha='right', rotation=45)
    plt.show()

def group(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return np.asarray((unique, counts)).T

plot_class_frequencies(train_labels, "Train")
plot_class_frequencies(test_labels, "Test")

#Best output
batch_size = 64
patch_size = 5
depth = 12
num_hidden1 = 64
num_hidden2 = 32
drop_out = 0.5

graph = tf.Graph()


####################################################
"""
The model architecture consists of 6 layers: 

1. Convolution Layer with 12 filters using 5X5 patch size applying a [1,1,1,1] filter using VALID padding
2. Pooling layer using Average Pooling with [1,2,2,1] filter using VALID padding
3. Convolution Layer with 12 filters using 5X5 patch size applying a [1,1,1,1] filter using VALID padding
4. Pooling layer using Average Pooling with [1,2,2,1] filter using VALID padding
5. Fully Connected Layer with 1452 hidden units and 0.5 dropout applied
6. Fully Connected Layer with 64 hidden units and 0.5 dropout and RELU applied
7. Fully Connected Layer with 32 hidden units and 0.5 dropout and RELU applied
8. Fully Connected Layer with 43 output nodes
"""

with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    size3 = ((image_size - patch_size + 1) // 2 - patch_size + 1) // 2
    layer3_weights = tf.Variable(tf.truncated_normal([size3 * size3 * depth, num_hidden1], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))
    layer5_weights = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
    def model(data, keep_prob):
        # C1
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
        bias1 = tf.nn.relu(conv1 + layer1_biases)
        # S2
        pool2 = tf.nn.avg_pool(bias1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # C3
        conv3 = tf.nn.conv2d(pool2, layer2_weights, [1, 1, 1, 1], padding='VALID')
        bias3 = tf.nn.relu(conv3 + layer2_biases)
        # S4
        pool4 = tf.nn.avg_pool(bias3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # F5 input 5 x 5
        shape = pool4.get_shape().as_list()
        reshape = tf.reshape(pool4, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden5 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        # F6
        drop5 = tf.nn.dropout(hidden5, keep_prob)
        hidden6 = tf.nn.relu(tf.matmul(hidden5, layer4_weights) + layer4_biases)
        drop6 = tf.nn.dropout(hidden6, keep_prob)
        #F7
        return tf.matmul(drop6, layer5_weights) + layer5_biases
  
  # Training computation.
    #if test == True:
    #    image_predictions = tf.nn.softmax(model(tf_test_images, 1.0))
    #else:
    logits = model(tf_train_dataset,drop_out)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))

#Best output
num_steps = 10001

#with tf.Session(graph=graph) as session:
session = tf.InteractiveSession(graph = graph)
with session.as_default():
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    saver = tf.train.Saver()
    model_path = get_model_dir()
    saver.save(session, '{}/the_model'.format(model_path))
    print('model_saved at {}/the_model'.format(model_path))

with open('new_test.p', 'rb') as n_file:
    test_images = pickle.load(n_file)

reshaped_images = rgb2gray(test_images)
mean = np.mean(reshaped_images)
reshaped_images /= mean
reshaped_images = reshaped_images.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

tf_test_images = tf.constant(reshaped_images)
outputs = tf.nn.softmax(model(tf_test_images, 1.0))
top_k = session.run(tf.nn.top_k(input=outputs.eval(), k=5, sorted=True))

top_labels = []
for labels in top_k.indices :
    top_labels.append(get_label_map('signnames.csv', labels))

_, axs = plt.subplots(2,3)
axs = axs.ravel()
plt.figure(figsize=(2,2));
for i, (test_image, label) in enumerate(zip(test_images, top_labels)):
    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])
    axs[i].imshow(test_image)
    axs[i].set_title(label[0])
    # plt.title(label[0])
plt.show();

#Visualizing the tf.nn.top_k softmax probability outputs as arrays
top_k

for i, (labels, probs, candidate) in enumerate(zip(top_k.indices, top_k.values, test_images)):
    fig = plt.figure(figsize=(15, 2))
    plt.bar(labels,probs)
    plt.title(top_labels[i][0])
    height = candidate.shape[0]
    plt.xticks(np.arange(0.5, 43.5, 1.0), get_label_map('signnames.csv', np.unique(train['labels'])),  ha='right', rotation=45)
    plt.yticks(np.arange(0.0,1.0,0.1), np.arange(0.0, 1.0, 0.1))
    ax = plt.axes([.75,0.25, 0.5, 0.5], frameon=True)  # Change the numbers in this array to position your image [left, bottom, width, height])
    ax.imshow(candidate)
    ax.axis('off')  # get rid of the ticks and ticklabels
    
plt.show()



