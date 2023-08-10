import numpy as np

map_func = dict(linear=lambda x: x, square=lambda x: x**2, sin=np.sin)

def score_func(x):
    score = np.sin(x["x2"]) + map_func[x["x4"]](x["x1"]) + map_func[x["x4"]](x["x3"])
    return score


params_conf = [
    {"name": "x1", "domain": (.1, 5), "type": "continuous",
     "num_grid": 5, "scale": "log"},
    {"name": "x2", "domain": (-5, 5), "type": "continuous",
     "num_grid": 5},
    {"name": "x3", "domain": (-5, 3), "type": "continuous",
     "num_grid": 5},
    {"name": "x4", "domain": ("linear", "sin", "square"),
     "type": "categorical"},
]

from bboptimizer import Optimizer
import random

np.random.seed(0)
random.seed(0)
bayes_opt = Optimizer(score_func, params_conf, sampler="bayes", r_min=10, maximize=True)
bayes_opt.search(num_iter=50)


np.random.seed(0)
random.seed(0)
random_opt = Optimizer(score_func, params_conf, sampler="random", maximize=True)
random_opt.search(num_iter=50)


np.random.seed(0)
random.seed(0)
grid_opt = Optimizer(score_func, params_conf, sampler="grid", num_grid=3, maximize=True)
grid_opt.search(num_iter=50)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20,10))
X = np.arange(1, len(bayes_opt.results[1]) + 1)
plt.plot(X, bayes_opt.results[1], color="b", label="bayes")
plt.plot(X, random_opt.results[1], color="g", label="random")
plt.plot(X, grid_opt.results[1], color="y", label="grid")

plt.scatter(X, bayes_opt.results[1], color="b")
plt.scatter(X, random_opt.results[1], color="g")
plt.scatter(X, grid_opt.results[1], color="y")

plt.xlabel("the number of trials", fontsize=30)
plt.ylabel("score", fontsize=30)
plt.title("Optimization results", fontsize=50)

plt.legend(fontsize=20)
plt.savefig("toy_model_opt.jpg")

print("bayes", bayes_opt.best_results)
print("random", random_opt.best_results)
print("grid", grid_opt.best_results)

from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


data, target = make_classification(n_samples=2500,
                                   n_features=45,
                                   n_informative=5,
                                   n_redundant=5)


def score_func(params):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_idx, test_idx = list(splitter.split(data, target))[0]
    train_data = data[train_idx]
    train_target = target[train_idx]
    clf = SVC(**params)
    clf.fit(train_data, train_target)
    pred = clf.predict(data[test_idx])
    true_y = target[test_idx]
    score = accuracy_score(true_y, pred)
    return score

params_conf = [
    {'name': 'C', 'domain': (1e-8, 1e5), 'type': 'continuous', 'scale': 'log'},
    {'name': 'gamma', 'domain': (1e-8, 1e5), 'type': 'continuous', 'scale': 'log'},
    {'name': 'kernel', 'domain': 'rbf', 'type': 'fixed'}
]

from bboptimizer import Optimizer
import random
import numpy as np

np.random.seed(0)
random.seed(0)
bayes_opt = Optimizer(score_func, params_conf, sampler="bayes", r_min=10, maximize=True)
bayes_opt.search(num_iter=50)


np.random.seed(0)
random.seed(0)
random_opt = Optimizer(score_func, params_conf, sampler="random", maximize=True)
random_opt.search(num_iter=50)


np.random.seed(0)
random.seed(0)
grid_opt = Optimizer(score_func, params_conf, sampler="grid", num_grid=7, maximize=True)
grid_opt.search(num_iter=50)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20,10))
X = np.arange(1, len(bayes_opt.results[1]) + 1)
plt.plot(X, bayes_opt.results[1], color="b", label="bayes")
plt.plot(X, random_opt.results[1], color="g", label="random")
plt.plot(X, grid_opt.results[1], color="y", label="grid")

plt.scatter(X, bayes_opt.results[1], color="b")
plt.scatter(X, random_opt.results[1], color="g")
plt.scatter(X, grid_opt.results[1], color="y")

plt.xlabel("the number of trials", fontsize=30)
plt.ylabel("score", fontsize=30)
plt.title("Hyperparameter Optimization", fontsize=50)

plt.legend(fontsize=20)
plt.savefig("hyper_opt.jpg")

print("bayes", bayes_opt.best_results)
print("random", random_opt.best_results)
print("grid", grid_opt.best_results)

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf

# Fetch MNIST dataset
mnist = tf.contrib.learn.datasets.load_dataset("mnist")


train = mnist.train
X = train.images
train_X = X
train_y = np.expand_dims(train.labels, -1)
train_y = OneHotEncoder().fit_transform(train_y)

valid = mnist.validation
X = valid.images
valid_X = X 
valid_y = np.expand_dims(valid.labels, -1)
valid_y = OneHotEncoder().fit_transform(valid_y)

test = mnist.test
X = test.images
test_X = X
test_y = test.labels

from bboptimizer import Optimizer
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers import Activation, Reshape
from keras.optimizers import Adam, Adadelta, SGD, RMSprop
from keras.regularizers import l1, l2


def get_optimzier(name, **kwargs):
    if name == "rmsprop":
        return RMSprop(**kwargs)
    elif name == "adam":
        return Adam(**kwargs)
    elif name == "sgd":
        return SGD(**kwargs)
    elif name == "adadelta":
        return Adadelta(**kwargs)
    else:
        raise ValueError(name)


def construct_NN(params):
    model = Sequential()
    model.add(Reshape((784,), input_shape=(784,)))
    
    def update_model(_model, _params, name):
        _model.add(Dropout(_params[name + "_drop_rate"]))
        _model.add(Dense(units=_params[name + "_num_units"],
                    activation=None,
                    kernel_regularizer=l2(_params[name + "_w_reg"])))
        if _params[name + "_is_batch"]:
            _model.add(BatchNormalization())
        if _params[name + "_activation"] is not None:
            _model.add(Activation(_params[name + "_activation"]))
        return _model
    
    # Add input layer    
    model = update_model(model, params, "input")
    # Add hidden layer
    for i in range(params["num_hidden_layers"]):
        model = update_model(model, params, "hidden")
    # Add output layer
    model = update_model(model, params, "output")
    optimizer = get_optimzier(params["optimizer"],
                              lr=params["learning_rate"])
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
        

def score_func(params):
    # print("parameters", params)
    model = construct_NN(params)
    model.fit(train_X, train_y,
              epochs=params["epochs"],
              batch_size=params["batch_size"], verbose=1)
    # print("###################", model.metrics_names)
    score = model.evaluate(valid_X, valid_y,
                  batch_size=params["batch_size"])
    idx = model.metrics_names.index("acc")
    score = score[idx]
    print(params, score)
    return score

params_conf = [
    {"name": "num_hidden_layers", "type": "integer",
     "domain": (0, 5)},
    {"name": "batch_size", "type": "integer",
     "domain": (16, 128), "scale": "log"},
    {"name": "learning_rate", "type": "continuous",
     "domain": (1e-5, 1e-1), "scale": "log"},
    {"name": "epochs", "type": "integer",
     "domain": (10, 250), "scale": "log"},
    {"name": "optimizer", "type": "categorical",
     "domain": ("rmsprop", "sgd", "adam", "adadelta")},
    
    {"name": "input_drop_rate", "type": "continuous",
     "domain": (0, 0.5)},
    {"name": "input_num_units", "type": "integer",
     "domain": (32, 512), "scale": "log"},
    {"name": "input_w_reg", "type": "continuous",
     "domain": (1e-10, 1e-1), "scale": "log"},
    {"name": "input_is_batch", "type": "categorical",
     "domain": (True, False)},
    {"name": "input_activation", "type": "categorical",
     "domain": ("relu", "sigmoid", "tanh")},
    
    {"name": "hidden_drop_rate", "type": "continuous",
     "domain": (0, 0.75)},
    {"name": "hidden_num_units", "type": "integer",
     "domain": (32, 512), "scale": "log"},
    {"name": "hidden_w_reg", "type": "continuous",
     "domain": (1e-10, 1e-1), "scale": "log"},
    {"name": "hidden_is_batch", "type": "categorical",
     "domain": (True, False)},
    {"name": "hidden_activation", "type": "categorical",
     "domain": ("relu", "sigmoid", "tanh")},
    
    {"name": "output_drop_rate", "type": "continuous",
     "domain": (0, 0.5)},
    {"name": "output_num_units", "type": "fixed",
     "domain": 10},
    {"name": "output_w_reg", "type": "continuous",
     "domain": (1e-10, 1e-1), "scale": "log"},
    {"name": "output_is_batch", "type": "categorical",
     "domain": (True, False)},
    {"name": "output_activation", "type": "fixed",
     "domain": "softmax"},
    
]



from bboptimizer import Optimizer
import random

np.random.seed(0)
random.seed(0)
bayes_opt = Optimizer(score_func, params_conf, sampler="bayes", r_min=10, maximize=True)
bayes_opt.search(num_iter=50)


np.random.seed(0)
random.seed(0)
random_opt = Optimizer(score_func, params_conf, sampler="random", maximize=True)
random_opt.search(num_iter=50)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20,10))
X = np.arange(1, len(bayes_opt.results[1]) + 1)
plt.plot(X, bayes_opt.results[1], color="b", label="bayes")
plt.plot(X, random_opt.results[1], color="g", label="random")
# plt.plot(X, grid_opt.results[1], color="y", label="grid")

plt.scatter(X, bayes_opt.results[1], color="b")
plt.scatter(X, random_opt.results[1], color="g")
# plt.scatter(X, grid_opt.results[1], color="y")

plt.xlabel("the number of trials", fontsize=30)
plt.ylabel("score", fontsize=30)
plt.title("Neural Network Hyperparameter Optimization", fontsize=50)

plt.ylim(0.96, 1.0)

plt.legend(fontsize=20)
plt.savefig("hyper_nn_opt.jpg")



