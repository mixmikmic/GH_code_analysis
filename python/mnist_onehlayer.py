import argparse
import math
import os
import time

from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# Define some constants.
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# Batch size. Must be evenly dividable by dataset sizes.
BATCH_SIZE = 100
EVAL_BATCH_SIZE = 3
# Number of units in hidden layers.
HIDDEN1_UNITS = 128

NUM_STEPS = 25000
DATA_DIR = "MNIST_data"
MODEL_DIR = default=os.path.join(
    "/tmp/tfmodels/mnist_onehlayer",
    str(int(time.time())))

# Build inference graph.
def mnist_inference(images, hidden1_units):
    """Build the MNIST model up to where it may be used for inference.
    Args:
        images: Images placeholder.
        hidden1_units: Size of the first hidden layer.
    Returns:
        logits: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden1, weights) + biases
    return logits

# Build training graph.
def mnist_training(logits, labels, learning_rate):
    """Build the training graph.

    Args:
        logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
        labels: Labels tensor, int32 - [BATCH_SIZE], with values in the
          range [0, NUM_CLASSES).
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, loss

def model_train_and_eval():
    """Build the full graph for feeding inputs, training, and
    saving checkpoints.  Run the training. Then, load the saved graph and
    run some predictions."""

    # Get input data: get the sets of images and labels for training,
    # validation, and test on MNIST.
    data_sets = read_data_sets(DATA_DIR, False)

    mnist_graph = tf.Graph()
    with mnist_graph.as_default():
        # Generate placeholders for the images and labels.
        images_placeholder = tf.placeholder(tf.float32)
        labels_placeholder = tf.placeholder(tf.int32)
        tf.add_to_collection("images", images_placeholder)  # Remember this Op.
        tf.add_to_collection("labels", labels_placeholder)  # Remember this Op.

        # Build a Graph that computes predictions from the inference model.
        logits = mnist_inference(images_placeholder,
                                 HIDDEN1_UNITS)
        tf.add_to_collection("logits", logits)  # Remember this Op.

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op, loss = mnist_training(
            logits, labels_placeholder, 0.01)

        # prediction accuracy
        _, indices_op = tf.nn.top_k(logits)
        flattened = tf.reshape(indices_op, [-1])
        correct_prediction = tf.cast(
            tf.equal(labels_placeholder, flattened), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # Define info to be used by the SummaryWriter. This will let
        # TensorBoard plot values during the training process.
        loss_summary = tf.scalar_summary("loss", loss)
        train_summary_op = tf.merge_summary([loss_summary])

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a summary writer.
        print("Writing Summaries to %s" % MODEL_DIR)
        train_summary_writer = tf.train.SummaryWriter(MODEL_DIR)

    # Run training for MAX_STEPS and save checkpoint at the end.
    with tf.Session(graph=mnist_graph) as sess:
        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        print("Starting training...")
        for step in xrange(NUM_STEPS):
            # Read a batch of images and labels.
            images_feed, labels_feed = data_sets.train.next_batch(BATCH_SIZE)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value, tsummary, acc = sess.run(
                [train_op, loss, train_summary_op, accuracy],
                feed_dict={images_placeholder: images_feed,
                           labels_placeholder: labels_feed})
            if step % 100 == 0:
                # Write summary info
                train_summary_writer.add_summary(tsummary, step)
            if step % 1000 == 0:
                # Print loss/accuracy info
                print('----Step %d: loss = %.4f' % (step, loss_value))
                print("accuracy: %s" % acc)

        print("\nFinished training. Writing checkpoint file.")
        checkpoint_file = os.path.join(MODEL_DIR, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)
        _, loss_value = sess.run(
            [train_op, loss],
            feed_dict={images_placeholder: data_sets.test.images,
                       labels_placeholder: data_sets.test.labels})
        print("Test set loss: %s" % loss_value)

    # Run evaluation based on the saved checkpoint.
    with tf.Session(graph=tf.Graph()) as sess:
        checkpoint_file = tf.train.latest_checkpoint(MODEL_DIR)
        print("\nRunning predictions based on saved checkpoint.")
        print("checkpoint file: {}".format(checkpoint_file))
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Retrieve the Ops we 'remembered'.
        logits = tf.get_collection("logits")[0]
        images_placeholder = tf.get_collection("images")[0]
        labels_placeholder = tf.get_collection("labels")[0]

        # Add an Op that chooses the top k predictions.
        eval_op = tf.nn.top_k(logits)

        # Run evaluation.
        images_feed, labels_feed = data_sets.validation.next_batch(
            EVAL_BATCH_SIZE)
        prediction = sess.run(eval_op,
                              feed_dict={images_placeholder: images_feed,
                                         labels_placeholder: labels_feed})
        for i in range(len(labels_feed)):
            print("Ground truth: %d\nPrediction: %d" %
                  (labels_feed[i], prediction.indices[i][0]))

model_train_and_eval()

