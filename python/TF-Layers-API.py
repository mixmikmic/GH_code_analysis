# Load Dataset
from sklearn.datasets import load_wine

wine_data = load_wine()

type(wine_data)

wine_data.keys()

print(wine_data['DESCR'])

feat_data = wine_data['data']

labels = wine_data['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feat_data, labels, test_size=0.3, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_x_train = scaler.fit_transform(X_train)

scaled_x_test = scaler.transform(X_test)

import tensorflow as tf
import pandas as pd

oneHot_y_train = pd.get_dummies(y_train).as_matrix()

oneHot_y_test = pd.get_dummies(y_test).as_matrix()

num_features = 13
hidden1 = 13
hidden2 = 13
output_layer = 3
lr = 0.01

from tensorflow.contrib.layers import fully_connected

X = tf.placeholder(tf.float32, shape=[None, num_features])
y = tf.placeholder(tf.float32, shape=[None,output_layer])

activation_fn = tf.nn.relu

hidden_layer1 = fully_connected(X, hidden1, activation_fn=activation_fn)
hidden_layer2 = fully_connected(hidden_layer1, hidden2, activation_fn=activation_fn)
output = fully_connected(hidden_layer2, output_layer) 

# Loss Function
loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits = output)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

train_steps = 3

# Save Trained Model
save_model = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    for steps in range(train_steps):
        sess.run(optimizer, feed_dict={X:scaled_x_train, y: oneHot_y_train})
    
    logits = output.eval(feed_dict={X:scaled_x_test})
    
    # Get value with highest probability prediction
    preds = tf.argmax(logits, axis=1)
    
    results = preds.eval()
    
    save_model.save(sess, './tf-layers-saved-model/tf-layers-trained-model')

from sklearn.metrics import classification_report

print(classification_report(results, y_test))

