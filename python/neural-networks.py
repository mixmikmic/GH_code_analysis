import zipfile
with zipfile.ZipFile("reviews.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

with open("reviews.txt") as f:
    reviews = f.read().split("\n")
with open("labels.txt") as f:
    labels = f.read().split("\n")
    
reviews_tokens = [review.split() for review in reviews]

from sklearn.preprocessing import MultiLabelBinarizer

onehot_enc = MultiLabelBinarizer()
onehot_enc.fit(reviews_tokens)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(reviews_tokens, labels, test_size=0.4, random_state=None)

split_point = int(len(X_test)/2)
X_valid, y_valid = X_test[split_point:], y_test[split_point:]
X_test, y_test = X_test[:split_point], y_test[:split_point]    

import tensorflow as tf

tf.reset_default_graph()

vocab_len = len(onehot_enc.classes_)
inputs_ = tf.placeholder(dtype=tf.float32, shape=[None, vocab_len], name="inputs")
targets_ = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="targets")

h1 = tf.layers.dense(inputs_, 500, activation=tf.nn.relu)
#h2 = tf.layers.dense(h1, 500, activation=tf.nn.relu)
logits = tf.layers.dense(h1, 2, activation=None)
output = tf.nn.sigmoid(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets_))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(targets_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def label2bool(labels):
    return [[1,0] if label == "positive" else [0,1] for label in labels]

def get_batch(X, y, batch_size):
    for batch_pos in range(0,len(X),batch_size):
        yield X[batch_pos:batch_pos+batch_size], y[batch_pos:batch_pos+batch_size]    

epochs = 10
batch_size = 3000

sess = tf.Session()

# Initializing the variables
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    for X_batch, y_batch in get_batch(onehot_enc.transform(X_train), label2bool(y_train), batch_size):         
        loss_value, _ = sess.run([loss, optimizer], feed_dict={
            inputs_: X_batch,
            targets_: y_batch
        })
        print("Epoch: {} \t Training loss: {}".format(epoch, loss_value))

    acc = sess.run(accuracy, feed_dict={
        inputs_: onehot_enc.transform(X_valid),
        targets_: label2bool(y_valid)
    })

    print("Epoch: {} \t Validation Accuracy: {}".format(epoch, acc))

test_acc = sess.run(accuracy, feed_dict={
    inputs_: onehot_enc.transform(X_test),
    targets_: label2bool(y_test)
})
print("Test Accuracy: {}".format(test_acc))

#Test model
sentence = "great movie"
sentence_tokens = onehot_enc.transform([sentence.split()])

result = sess.run(output, feed_dict={
    inputs_: sentence_tokens
})
if result[0][0] > result[0][1]:
    print("positive")
else:
    print("negative")



