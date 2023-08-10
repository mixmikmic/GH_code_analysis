import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import sys
seed = 782
np.random.seed(seed)

df = pd.read_csv("digit_train.csv")
train = df.as_matrix()

train_y = train[:30000,0].astype('int8')
train_x = train[:30000,1:].astype('float64')

test_y = train[30000:,0].astype('int8')
test_x = train[30000:,1:].astype('float64')


print("Shape Train Images: (%d,%d)" % train_x.shape)
print("Shape Labels: (%d)" % train_y.shape)
print("Shape Train Images: (%d,%d)" % test_x.shape)
print("Shape Labels: (%d)" % test_y.shape)

# df = pd.read_csv("digit_test.csv")
# test = df.as_matrix().astype('float64')
# print("Shape Test Images: (%d,%d)" % test.shape)

def show_image(image, shape, label="", cmp=None):
    img = np.reshape(image,shape)
    plt.imshow(img,cmap=cmp, interpolation='none')
    plt.title(label)

# np.random.randint(0,train_x.shape[0],1)[0]

get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,10))

y, x = 5,10
for i in range(0,(y*x)):
    plt.subplot(y, x, i+1)
    ni = np.random.randint(0,train_x.shape[0],1)[0]
    show_image(train_x[ni],(28,28), train_y[ni], cmp="gray")
plt.show()

def count_exemple_per_digit(exemples):
    hist = np.ones(10)

    for y in exemples:
        hist[y] += 1

    colors = []
    for i in range(10):
        colors.append(plt.get_cmap('viridis')(np.random.uniform(0.0,1.0,1)[0]))

    bar = plt.bar(np.arange(10), hist, 0.8, color=colors)

    plt.grid()
    plt.show()

count_exemple_per_digit(train_y)

# from sklearn.linear_model import LogisticRegression
# logregressor = LogisticRegression(solver="liblinear", multi_class="ovr")

# train_y = df_labels["label"].as_matrix()  # To get the right vector-like shape call as_matrix on the single column
# train_X = df_images.as_matrix()
# logregressor.fit(train_X, train_y)

# from sklearn.datasets import fetch_mldata

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(train_x/255., train_y)

print("Training set score: %f" % mlp.score(train_x, train_y))
print("Test set score: %f" % mlp.score(test_x, test_y))

# use below code to reconstruct the learned model 
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

from sklearn.model_selection import GridSearchCV

df = pd.read_csv("digit_train.csv")
train = df.as_matrix()

param_grid = {
        'hidden_layer_sizes': [(50,)],
        'tol': [1e-2, 1e-3],
        'epsilon': [1e-3, 1e-7, 1e-8, 1e-9, 1e-8]
    }

estimator = GridSearchCV(
        MLPClassifier(learning_rate='constant', learning_rate_init=.1, early_stopping=True, shuffle=True),
        param_grid=param_grid, n_jobs=-1)


# train_y = train[:,0].astype('int8')
# train_x = train[:,1:].astype('float64')

estimator.fit(train_x/255., train_y)

print (estimator.score(test_x, test_y))
print (estimator.best_params_ , estimator.best_score_)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

pred = estimator.predict(test_x)
accuracy_score(pred,test_y)
confusion_matrix(pred,test_y)

print(classification_report(pred,test_y))



