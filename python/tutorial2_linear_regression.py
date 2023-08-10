# All imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import get_log_path
import pandas as pd
import shutil
get_ipython().magic('matplotlib inline')

df = pd.read_excel('data/fire_theft.xls')
my_data = df.as_matrix()
df.head()

df.describe()

class Config():
    """
    Class to hold all model hyperparams.
    :type learning_rate: float
    :type delta: float
    :type huber: boolean
    :type num_epochs: int
    :type show_epoch: int
    :type log_path: None or str
    """
    def __init__(self,
                 learning_rate=0.001,
                 delta=1.0,
                 huber=False,
                 num_epochs=101,
                 show_epoch=10,
                 log_path=None):
        self.learning_rate = learning_rate
        self.delta = delta
        self.huber = huber
        self.num_epochs = num_epochs
        self.show_epoch = show_epoch
        if log_path is None:
            self.log_path = get_log_path()
        else:
            self.log_path = log_path

class LinearRegression:
    """
    Class for the linear regression model
    
    :type config: Config
    """
    def __init__(self, config):
        self.learning_rate = config.learning_rate
        self.delta = config.delta
        self.huber = config.huber
        self.log_path = config.log_path
        self.build_graph()

    def create_placeholders(self):
        """
        Method for creating placeholders for input X (number of fire)
        and label Y (number of theft).
        """
        self.number_fire = tf.placeholder(tf.float32, shape=[], name="X")
        self.number_theft = tf.placeholder(tf.float32, shape=[], name="Y")

    def create_variables(self):
        """
        Method for creating weight and bias variables.
        """
        with tf.name_scope("Weights"):
            self.weight = tf.get_variable("w", dtype=tf.float32, initializer=0.)
            self.bias = tf.get_variable("b", dtype=tf.float32, initializer=0.)

    def create_summaries(self):
        """
        Method to create the histogram summaries for all variables
        """
        tf.summary.histogram('weights_summ', self.weight)
        tf.summary.histogram('bias_summ', self.bias)

    def create_prediction(self):
        """
        Method for creating the linear regression prediction.
        """
        with tf.name_scope("linear-model"):
            self.prediction = (self.number_fire * self.weight) + self.bias

    def create_MSE_loss(self):
        """
        Method for creating the mean square error loss function.
        """
        with tf.name_scope("loss"):
            self.loss = tf.square(self.prediction - self.number_theft)
            tf.summary.scalar("loss", self.loss)

    def create_Huber_loss(self):
        """
        Method for creating the Huber loss function.
        """
        with tf.name_scope("loss"):
            residual = tf.abs(self.prediction - self.number_theft)
            condition = tf.less(residual, self.delta)
            small_residual = 0.5 * tf.square(residual)
            large_residual = self.delta * residual - 0.5 * tf.square(self.delta)
            self.loss = tf.where(condition, small_residual, large_residual)
            tf.summary.scalar("loss", self.loss)

    def create_optimizer(self):
        """
        Method to create the optimizer of the graph
        """
        with tf.name_scope("optimizer"):
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.optimizer = opt.minimize(self.loss)

    def build_graph(self):
        """
        Method to build the computation graph in tensorflow
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_placeholders()
            self.create_variables()
            self.create_summaries()
            self.create_prediction()
            if self.huber:
                self.create_Huber_loss()
            else:
                self.create_MSE_loss()
            self.create_optimizer()

def run_training(model, config, data, verbose=True):
    """
    Function to train the linear regression model

    :type model: LinearRegression
    :type config: Config
    :type data: np array
    :type verbose: boolean
    :rtype total_loss: float
    :rtype w: float
    :rtype b: float
    """
    num_samples = data.shape[0]
    num_epochs = config.num_epochs
    show_epoch = config.show_epoch
    log_path = model.log_path
    with tf.Session(graph=model.graph) as sess:
        if verbose:
            print('Start training\n')
        # functions to write the tensorboard logs
        summary_writer = tf.summary.FileWriter(log_path,sess.graph)
        all_summaries = tf.summary.merge_all()
        # initializing variables
        tf.global_variables_initializer().run()
        step = 0
        for i in range(num_epochs): # run num_epochs epochs
            total_loss = 0
            for x, y in data:
                step += 1
                
                feed_dict = {model.number_fire: x,
                             model.number_theft: y}
                
                _,loss,summary,w,b = sess.run([model.optimizer, # run optimizer to perform minimization
                                               model.loss,
                                               all_summaries,
                                               model.weight,
                                               model.bias], feed_dict=feed_dict)

                #writing the log
                summary_writer.add_summary(summary,step)
                summary_writer.flush()
                
                total_loss += loss
            if i % show_epoch == 0:
                print("\nEpoch {0}: {1}".format(i, total_loss/num_samples))
    if verbose:
        print("\n========= For TensorBoard visualization type ===========")
        print("\ntensorboard  --logdir={}\n".format(log_path))
    return total_loss,w,b 

my_config = Config()
my_model = LinearRegression(my_config)
l,w,b = run_training(my_model, my_config, my_data)

# !tensorboard  --logdir=

def lr_tunning(data, number_of_exp=10, clean=True, huber=False):
    """
    Function that returns the best weights after training the model
    with some random values for the learning rate. 
    
    :type data: np array
    :type number_of_exp: int
    :type clean: boolean
    :type huber: boolean
    :rtype w: float
    :rtype b: float
    """
    if clean:
        shutil.rmtree("./graphs")
    num_samples = data.shape[0]
    LR = np.random.random_sample([number_of_exp])/1000
    LR.sort()
    best_loss = float('inf')
    for i, lr in enumerate(LR):
        log_path = './graphs/' + str(lr)
        header1 = "\n=============== ({0} of {1}) ===============\n".format(i + 1, number_of_exp)
        header2 = "  learning rate = {}".format(lr)
        header3 = "\n=========================================\n"
        print(header1 + header2 + header3)
        my_config = Config(log_path=log_path,
                           learning_rate=lr,
                           show_epoch=100,
                           huber=huber)
        my_model = LinearRegression(my_config)
        current_loss, current_w, current_b = run_training(my_model,
                                                          my_config,
                                                          data,
                                                          verbose=False)
        if current_loss < best_loss:
            best_loss, best_lr = current_loss, lr
            w, b = current_w, current_b
    print(header3)
    print("\nbest learning rate = {0}\nbest loss = {1}".format(best_lr,
                                                               best_loss/num_samples))
    print("\nFor TensorBoard visualization type")
    print("\ntensorboard  --logdir=/graphs/\n")
    return w, b

w,b = lr_tunning(my_data)

# !tensorboard --logdir=./graphs

def r_squared(data, w, b):
    """
    Calculate the R^2 value
    
    :type data: np array
    :type w: float
    :type b: float
    :rtype: float
    """
    X, Y = data.T[0], data.T[1]
    Y_hat = X * w + b
    Y_mean = np.mean(Y) 
    sstot = np.sum(np.square(Y - Y_mean))
    ssreg = np.sum(np.square(Y_hat - Y_mean))
    return 1 - (ssreg/sstot)

def plot_line(data, w, b, title, r_squared):
    """
    Plot the regression line
    
    :type data: np array
    :type w: float
    :type b: float
    :type title: str
    :type r_squared: float
    """
    X, Y = data.T[0], data.T[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(X, Y, 'bo', label='Real data')
    plt.plot(X, X * w + b, 'r', label='Predicted data')
    plt.title(title)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.2)
    t = ax.text(20, 135, "$R^2 ={:.4f}$".format(r_squared), size=15, bbox=bbox_props)
    plt.legend()
    plt.show()

r2 = r_squared(my_data,w,b)
plot_line(my_data, w, b, "Linear Regression with MSE", r2)

w,b = lr_tunning(my_data,huber=True)

r2 = r_squared(my_data,w,b)
plot_line(my_data, w, b, "Linear Regression with Huber loss", r2)



