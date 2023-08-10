import itertools

import numpy as np
import pandas as pd

import sklearn.datasets
import sklearn.neighbors
import sklearn.model_selection

import bokeh.plotting
import bokeh.layouts
import bokeh.io

bokeh.io.output_notebook()

def rgba_from_4bit(img_4):
    n, m = img_4.shape
    img_rgba = np.empty((n, m), dtype=np.uint32)
    view = img_rgba.view(dtype=np.uint8).reshape((n, m, 4))
    view[:, :, 3] = 255  # set all alpha values to fully visible
    rgba = 255 - img_4[:, :] / 16 * 255
    
    # rgba is upside-down, hence the ::-1
    view[:, :, 0] = view[:, :, 1] = view[:, :, 2] = rgba[::-1]
    
    return img_rgba

digits = sklearn.datasets.load_digits()

print(digits.DESCR)

p = bokeh.plotting.figure(width=110, height=100, x_range=(0, 8), y_range=(0, 8),
                          tools='', title='Training: {}'.format(digits.target[0]))
p.xaxis.visible = p.yaxis.visible = False

p.image_rgba(image=[rgba_from_4bit(digits.images[0])], x=0, y=0, dw=8, dh=8)

bokeh.plotting.show(p)

n_plot = 4
plots = []
w = 80
h = 80

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:n_plot]):
    img_rgba = rgba_from_4bit(image)
    p = bokeh.plotting.figure(width=w, height=h, tools='', 
                              x_range=(0, 8), y_range=(0, 8),
                              title='Training: {}'.format(label))
    p.xaxis.visible = p.yaxis.visible = False
    p.image_rgba(image=[img_rgba], x=0, y=0, dw=8, dh=8)
    plots.append(p)

grid = bokeh.layouts.gridplot([plots])
bokeh.plotting.show(grid)

X, y = digits.data, digits.target

classifier = sklearn.neighbors.KNeighborsClassifier()

train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y,
                                                                            train_size=0.8,
                                                                            random_state=123)
print('Without stratification:')
print('All:\n', np.bincount(y) / float(len(y)) * 100.0)
print('Training:\n', np.bincount(train_y) / float(len(train_y)) * 100.0)
print('Test:\n', np.bincount(test_y) / float(len(test_y)) * 100.0)

train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y,
                                                                            train_size=0.8,
                                                                            random_state=123,
                                                                            stratify=y)
print('With stratification:')
print('All:\n', np.bincount(y) / float(len(y)) * 100.0)
print('Training:\n', np.bincount(train_y) / float(len(train_y)) * 100.0)
print('Test:\n', np.bincount(test_y) / float(len(test_y)) * 100.0)

classifier.fit(train_X, train_y)
pred_y = classifier.predict(test_X)

print("Fraction Correct [Accuracy]:")
print(np.sum(pred_y == test_y) / float(len(test_y)))

print('Samples correctly classified:')
correct_idx = np.where(pred_y == test_y)[0]
print(correct_idx)

print('\nSamples incorrectly classified:')
incorrect_idx = np.where(pred_y != test_y)[0]
print(incorrect_idx)

plots = []
w = 80
h = 80

for i in incorrect_idx:
    image = test_X[i].reshape(8, 8)
    img_rgba = rgba_from_4bit(image)
    p = bokeh.plotting.figure(width=w, height=h, tools='', 
                              x_range=(0, 8), y_range=(0, 8),
                              title='P: {}, A: {}'.format(pred_y[i], test_y[i]))
    p.xaxis.visible = p.yaxis.visible = False
    p.image_rgba(image=[img_rgba], x=0, y=0, dw=8, dh=8)
    plots.append(p)

grid = bokeh.layouts.gridplot([plots])
print("Incorrect Predictions:")
bokeh.plotting.show(grid)

import sklearn.cluster
import sklearn.metrics

kmeans = sklearn.cluster.KMeans(n_clusters=10, random_state=123)
labels = kmeans.fit_predict(X)

labels

accuracy = sklearn.metrics.adjusted_rand_score(y, labels)
print('Accuracy score:', accuracy)

confusion_matrix = sklearn.metrics.confusion_matrix(y, labels)
print(confusion_matrix)

key = {}
for i, r in enumerate(confusion_matrix):
    key[r.argmax()] = i
    print('group {} should be {}'.format(r.argmax(), i))

key = {}
for i in range(10):
    key[i] = confusion_matrix[:, i].argmax()
    print('group {} should be {}'.format(i, key[i]))

key

sorted_matrix = np.empty_like(confusion_matrix)
for i in range(10):
    sorted_matrix[:, key[i]] = confusion_matrix[:, i]
print(sorted_matrix)

sorted_labels = np.empty_like(labels)
for i, val in enumerate(labels):
    sorted_labels[i] = key[val]

sorted_confusion_matrix = sklearn.metrics.confusion_matrix(y, sorted_labels)
print(sorted_confusion_matrix)

n_p = np.array([(sorted_labels == i).sum() for i in range(10)])
n_a = np.array([(y == i).sum() for i in range(10)])
print('number of each:   {}'.format(n_a))
print('number predicted: {}'.format(n_p))

sorted_confusion_matrix.sum(axis=1)

sorted_confusion_matrix.sum(axis=0)

p_pos = bokeh.plotting.figure(width=200, height=200, x_range=(0, 10), y_range=(0, 10),
                              tools='', title='Confusion Matrix')
p_neg = bokeh.plotting.figure(width=200, height=200, x_range=(0, 10), y_range=(0, 10),
                              tools='')

p_pos.xaxis.visible = p_pos.yaxis.visible = False
p_neg.xaxis.visible = p_neg.yaxis.visible = False
img_pos = rgba_from_4bit(sorted_confusion_matrix * 16 / sorted_confusion_matrix.max())
img_neg = rgba_from_4bit((sorted_confusion_matrix * 16 / sorted_confusion_matrix.max() - 16) * -1)

p_pos.image_rgba(image=[img_pos], x=0, y=0, dw=10, dh=10)
p_neg.image_rgba(image=[img_neg], x=0, y=0, dw=10, dh=10)

plots = bokeh.layouts.gridplot([[p_pos, p_neg]])
bokeh.plotting.show(plots)



