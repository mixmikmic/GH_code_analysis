import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn as skflow
get_ipython().magic('pylab inline')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot=True)

#Display 9 random images from data set
def draw_image(x, y, title_str, drawTest = False):
    for c in range(1, 10):
        subplot(3, 3,c)
        i = randint(x.shape[0]) 
        im = x[i].reshape((28,28)) 
        axis("off")
        
        if not drawTest:
            label = np.argmax(y[i]) 
        else:
            label = y[i]
        title("{} = {}".format(title_str, label))
        imshow(im)

x_train = mnist.train.images
y_train = mnist.train.labels
x_validation = mnist.validation.images
y_validation = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels

print ("Training set: ", x_train.shape)
print ("Validation set: ", x_validation.shape)
print ("Test set: ", x_test.shape)

random.seed(42)
# Display 9 number randomly selectly
draw_image(mnist.train.images, mnist.train.labels, "Label")

# Building  neural network
model = tf.contrib.learn.DNNClassifier(feature_columns=  tf.contrib.learn.infer_real_valued_columns_from_input(x_train),
    hidden_units=[256, 256], 
    n_classes = 10,
    model_dir='model/')


model.fit(x=x_train, y=np.argmax(y_train,1), batch_size= 100, steps = 1000)

pred = model.predict(x_test)

# Calculate accuracy
with tf.Session() as sess:
    correct_prediction = tf.equal(pred, tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval())
    

draw_image(mnist.test.images, pred, "Pred",  True)

