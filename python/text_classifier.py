import numpy as np
import tensorflow as tf

from data_helpers import load_data_and_labels

X_sentences,y=load_data_and_labels('./data/rt/rt-polarity.pos','./data/rt/rt-polarity.neg')

X_sentences[0],y[0]

X_sentences[6338],y[6338]

list(map(len,X_sentences))[:20]

from tflearn.data_utils import VocabularyProcessor

sentence_len=60

vocab_proc = VocabularyProcessor(sentence_len)

X = np.array(list(vocab_proc.fit_transform(X_sentences)))

X[0]

next(vocab_proc.transform(['the reviews are']))

vocab_size = len(vocab_proc.vocabulary_)
#vocab_dict = vocab_processor.vocabulary_._mapping

#Global hyper-parameters
emb_dim=100
hidden_dim=50
num_classes=2

input_x = tf.placeholder(tf.int32, shape=[None, sentence_len], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
#dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

with tf.name_scope("embedding"):
    W = tf.Variable(tf.random_uniform([vocab_size, emb_dim], -1.0, 1.0),name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)

with tf.name_scope("reshape"):
    emb_vec= tf.reshape(embedded_chars,shape=[-1,sentence_len*emb_dim])

with tf.name_scope("hidden"):
    W_h= tf.Variable(tf.random_uniform([sentence_len*emb_dim, hidden_dim], -1.0, 1.0),name="w_hidden")
    b_h= tf.Variable(tf.zeros([hidden_dim],name="b_hidden"))
    hidden_output= tf.nn.relu(tf.matmul(emb_vec,W_h)+b_h)

with tf.name_scope("output_layer"):
    W_o= tf.Variable(tf.random_uniform([hidden_dim,2], -1.0, 1.0),name="w_o")
    b_o= tf.Variable(tf.zeros([2],name="b_o"))
    score = tf.nn.relu(tf.matmul(hidden_output,W_o)+b_o)
    predictions = tf.argmax(score, 1, name="predictions")

with tf.name_scope("loss"):
    losses=tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=score)
    loss=tf.reduce_mean(losses)

with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer=tf.train.AdamOptimizer(1e-4).minimize(loss)
loss_summary = tf.summary.scalar("loss", loss)
acc_summary = tf.summary.scalar("accuracy", accuracy)
summary_op=tf.summary.merge([loss_summary,acc_summary])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_summary_writer = tf.summary.FileWriter('./summaries/', sess.graph)

for i in range(100):
    acc,loss_,_=sess.run([accuracy,loss,optimizer],feed_dict={input_x:X,input_y:y})
    step,summaries = sess.run([global_step,summary_op],feed_dict={input_x:X,input_y:y})
    train_summary_writer.add_summary(summaries, i)
    print("This is step: %d, acc=%.2f, loss=%.2f"%(i,acc,loss_),end='\r')



