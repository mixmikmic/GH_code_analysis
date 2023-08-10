import numpy as np
import tensorflow as tf
import random
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

class WaveformData(object):
    def __init__(self, filename, anomaly_class, val_size):
        random.seed(42)
        f = open(filename,"r")
        data = f.read()
        lines = data.split('\n')

        self.data = []
        for line in lines[:-1]:
            self.data.append([float(x) for x in line.split(',')])
        self.val_size = val_size
        
        random.shuffle(self.data)
        self.anomaly_data = [x[:-1] for x in self.data if x[-1] == anomaly_class][:val_size]
        classes = [x[-1] for x in self.data if x[-1] != anomaly_class]
        self.data = [x[:-1] for x in self.data if x[-1] != anomaly_class]
        self.normal_val_data = self.data[:val_size]
        self.normal_classes = classes[:val_size]
        self.data = self.data[val_size:]
        self.classes = classes[val_size:]
        
    def normalize(self):
        # Normalize based on our training data
        self.mean, self.std = np.mean(self.data), np.std(self.data)
        self.data = (self.data - self.mean) / self.std
        self.normal_val_data = (self.normal_val_data - self.mean) / self.std
        self.anomaly_data = (self.anomaly_data - self.mean) / self.std

data = WaveformData('waveform.data', 2.0, 300)
data.normalize()

for i in range(6):
    if data.classes[i] == 0.:
        plt.plot(data.data[i], 'r')
    else:
        plt.plot(data.data[i], 'b')
for i in range(4):
    plt.plot(data.anomaly_data[i], 'g')

class Encoder(object):
    def __init__(self, inp, n_features, n_hidden, repr_size):
        # inp is the placeholder for the input, n_features is the number of features our data has (21 in this example)
        # n_hidden is the size of the first hidden layer and n_hidden_2 is the dimensionality of the representation
        self.inp = inp
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.repr_size = repr_size
        self.W1 = tf.Variable(tf.random_normal([n_features, n_hidden], stddev=0.35))
        self.W2 = tf.Variable(tf.random_normal([n_hidden, repr_size], stddev=0.35))
        self.b1 = tf.Variable(tf.random_normal([n_hidden], stddev=0.35))
        self.b2 = tf.Variable(tf.random_normal([repr_size], stddev=0.35))
        
        self.layer_1 = tf.nn.tanh(tf.matmul(self.inp, self.W1) + self.b1)
        self.encoder_out = tf.matmul(self.layer_1, self.W2) + self.b2

class Decoder(object):
    def __init__(self, inp, repr_size, n_hidden, n_features):
        self.inp = inp
        self.n_hidden = n_hidden
        self.W1 = tf.Variable(tf.random_normal([repr_size, n_hidden], stddev=0.35))
        self.W2 = tf.Variable(tf.random_normal([n_hidden, n_features], stddev=0.35))
        self.b1 = tf.Variable(tf.random_normal([n_hidden], stddev=0.35))
        self.b2 = tf.Variable(tf.random_normal([n_features], stddev=0.35))
        
        self.layer_1 = tf.nn.tanh(tf.matmul(self.inp, self.W1) + self.b1)
        self.decoder_out = tf.matmul(self.layer_1, self.W2) + self.b2

class Autoencoder(object):
    def __init__(self, n_features, repr_size, n_hidden, batch_size=16):
        # n_features is the number of features our data has (21 in this example)
        # repr_size the dimensionality of our representation
        # n_hidden_1 is the size of the layers closest to the in and output
        # n_hidden_2 is the size of the layers closest to the embedding layer
        # batch_size number of samples to run per batch
        
        self.n_features = n_features
        self.repr_size = repr_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        
        # Start session, placeholder has None in shape for batches
        self.sess = tf.Session()
        self.inp = tf.placeholder(tf.float32, [None, n_features])
        
        # Make the encoder and the decoder
        self.encoder = Encoder(self.inp, n_features, n_hidden, repr_size)
        self.decoder = Decoder(self.encoder.encoder_out, repr_size, n_hidden, n_features)
        
        # Loss function mean squared error and AdamOptimizer
        self.loss = tf.reduce_mean(tf.square(self.decoder.decoder_out - self.inp), -1)
        self.mean_loss = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.mean_loss)
        
        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())
        
    def run_epoch(self, data_list):
        # Train once over the passed data_list and return the mean reconstruction loss after the epoch
        for index in range(len(data.data) // self.batch_size):
            self.sess.run(self.train_op, feed_dict={self.inp: data_list[index * self.batch_size : (index+1): self.batch_size]})
        return self.sess.run(self.mean_loss, feed_dict={self.inp: data_list})
    
    def representations(self, data_list):
        # Return a list of representations for the given list of time series
        return self.sess.run(self.encoder.encoder_out, feed_dict={self.inp: data_list})
    
    def reconstruction_errors(self, data_list):
        # Get mean squared reconstruction errors of passed data_list
        return self.sess.run(self.loss, feed_dict={self.inp: data_list})

ae = Autoencoder(21, 2, 18)
for i in range(100):
    ae.run_epoch(data.data)

print(np.mean(ae.reconstruction_errors(data.data)))
ae.reconstruction_errors(data.data[:10])

print(np.mean(ae.reconstruction_errors(data.normal_val_data)))
ae.reconstruction_errors(data.normal_val_data[:10])

print(np.mean(ae.reconstruction_errors(data.anomaly_data)))
ae.reconstruction_errors(data.anomaly_data[:10])

plt.hist(ae.reconstruction_errors(data.anomaly_data), alpha=0.5, color='r')
plt.hist(ae.reconstruction_errors(data.normal_val_data), alpha=0.5, color='b')
plt.show()

anomaly_repr = ae.representations(data.anomaly_data)
normal_repr = ae.representations(data.normal_val_data)
anom_x, anom_y = zip(*anomaly_repr)
norm_x, norm_y = zip(*normal_repr)
plt.scatter(anom_x, anom_y, color='red', alpha=0.7)
plt.scatter(norm_x, norm_y, alpha=0.7)

from sklearn.metrics import roc_curve, auc

anomaly_errors = ae.reconstruction_errors(data.anomaly_data)
normal_val_errors = ae.reconstruction_errors(data.normal_val_data)
roc_y = [1 for _ in range(len(anomaly_errors))] + [0 for _ in range(len(normal_val_errors))]
roc_score = np.concatenate([anomaly_errors, normal_val_errors])

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(roc_y, roc_score)
roc_auc = auc(fpr, tpr)

# Stolen from plot_roc sklearn example at http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

