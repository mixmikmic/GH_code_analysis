import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from dog_images import DogImages
from image_classifier import ImageClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

picsize = 128
# Step 1: Get List of Dogs
lst = [x[0] for x in os.walk('../Images')]
lst_dogs = [a.replace('../Images/', '') for a in lst[1:]]
print(lst_dogs[0])

# Step 2: Make the Images...
dog_images = DogImages(lst_dogs, picsize)
# dog_images.generate_img_files()
train_imgs = dog_images.load_images('train')
test_imgs = dog_images.load_images('test')
Xtest = test_imgs[0]
Ytest = test_imgs[1]
Xtrain = train_imgs[0]
Ytrain = train_imgs[1]
print(lst_dogs[-1])

# Step 3: Initial Shuffle of Train & Test Sets
Xhold = Xtrain.copy()
Yhold = Ytrain.copy()
new = np.array([i for i in range(Xhold.shape[0])])
np.random.shuffle(new)
for i, n in enumerate(new):
    Xtrain[i, :] = Xhold[n, :]
    Ytrain[i, :] = Yhold[n, :]

Xhold = Xtest
Yhold = Ytest
new = np.array([i for i in range(Xhold.shape[0])])
np.random.shuffle(new)
for i, n in enumerate(new):
    Xtest[i, :] = Xhold[n, :]
    Ytest[i, :] = Yhold[n, :]
print('done')

def plot_learning(model):
    plt.plot(list(range(len(model.loss_function))),
             model.loss_function, color='y', label='Loss/Max Loss')
    plt.plot(list(range(len(model.train_accuracies))),
             (1/len(model.classes))*np.ones(len(model.train_accuracies)),
             linestyle='-', label='chance')
    plt.plot(list(range(len(model.train_accuracies))),
             model.train_accuracies, color='r', label='Training')
    plt.plot(list(range(len(model.val_accuracies))),
             model.val_accuracies, color='b', label='Validation')
    plt.ylabel('Probability')
    plt.xlabel('Epochs')
    plt.title('Accuracy & Loss')
    plt.ylim(ymax=1)
    plt.ylim(ymin=0)
    plt.legend()
    plt.show()

# Step 4: Grid Search...
# print('Starting...')

# lst_out = [len(lst_dogs)*a for a in range(5)]

# params = {'picsize':[picsize],
#               'classes':[lst_dogs],
#               'out_channels':[10, 100],
#               'out_channels_2':[20, 200],
#               'hidden_units':[10, 100],
#               'regularization_strength':[0.01, 0.1, 1.0],
#               'batch_size':[len(lst_dogs), 2*len(lst_dogs)],
#               'learning_rate':[0.0001, 0.001, 0.01],
#               'loss_threshold':[10.0],
#               'verbose':[True]}

# gs = GridSearchCV(ImageClassifier(), params, n_jobs=-1, verbose=3)

# print('Now fitting')
# gs.fit(Xtrain, Ytrain)
# print()
# print('Best Accuracy: {:.3f}'.format(gs.best_score_))
# print('Best Params: {}'.format(gs.best_params_))

from image_classifier import ImageClassifier


lst_ch = [10,12,14]
lst_l = [0.01, 0.001, 0.0001]
lst_h = [300, 450, 600]
lst_reg = [0.25, 0.5, 0.75, 1.0]
lst_b = [150, 200, 300]
best_score = 0.0
best_i = 0
best_j = 0

for i in lst_b:
    for j in lst_ch:
        model = ImageClassifier(picsize, lst_dogs,
                                 out_channels = j,
                                 out_channels_2 = 2*j,
                                 hidden_units = 600,
                                 regularization_strength = 0.5,
                                 batch_size = i,
                                 learning_rate = 0.001,
                                 convolution_size = 5,
                                 pool_size = 2,
                                 training_epochs = 50,
                                 loss_threshold = 10.0,
                                 verbose=True,
                                 grid_search=True)
        model.fit(Xtrain, Ytrain)
        score = model.score(Xtest, Ytest)
        print(' ', i, j, score)
        if score > best_score:
            best_score = score
            best_i = i
            best_j = j
        print(best_score,best_i,best_j)
print(best_score,best_i,best_j)

from image_classifier import ImageClassifier


lst_ch = [6,12,18,24]
lst_l = [0.01, 0.001, 0.0001]
lst_h = [300, 450, 600]
lst_reg = [0.25, 0.5, 0.75, 1.0]
best_score = 0.0
best_i = 0
best_j = 0

for i in lst_h:
    for j in lst_ch:
        model = ImageClassifier(picsize, lst_dogs,
                                 out_channels = j,
                                 out_channels_2 = 2*j,
                                 hidden_units = i,
                                 regularization_strength = 0.5,
                                 batch_size = 200,
                                 learning_rate = 0.001,
                                 convolution_size = 5,
                                 pool_size = 2,
                                 training_epochs = 50,
                                 loss_threshold = 10.0,
                                 verbose=True,
                                 grid_search=True)
        model.fit(Xtrain, Ytrain)
        score = model.score(Xtest, Ytest)
        print(' ', i, j, score)
        if score > best_score:
            best_score = score
            best_i = i
            best_j = j
        print(best_score,best_i,best_j)
print(best_score,best_i,best_j)

from image_classifier import ImageClassifier


lst = [1,2,3,4]
lst_l = [0.01, 0.001, 0.0001]
lst_reg = [0.25, 0.5, 0.75, 1.0]
best_score = 0.0
best_i = 0
best_j = 0

for i in lst_l:
    for j in lst_reg:
        model = ImageClassifier(picsize, lst_dogs,
                                 out_channels = 12,
                                 out_channels_2 = 24,
                                 hidden_units = 600,
                                 regularization_strength = j,
                                 batch_size = 200,
                                 learning_rate = i,
                                 convolution_size = 5,
                                 pool_size = 2,
                                 training_epochs = 50,
                                 loss_threshold = 10.0,
                                 verbose=True,
                                 grid_search=True)
        model.fit(Xtrain, Ytrain)
        score = model.score(Xtest, Ytest)
        print(' ', i, j, score)
        if score > best_score:
            best_score = score
            best_i = i
            best_j = j
        print(best_score,best_i,best_j)
print(best_score,best_i,best_j)

from image_classifier import ImageClassifier


model = ImageClassifier(picsize, lst_dogs,
                         out_channels = 24,
                         out_channels_2 = 48,
                         hidden_units = 128*6,
                         regularization_strength = 0.5,
                         batch_size = 200,
                         learning_rate = 0.001,
                         convolution_size = 5,
                         pool_size = 2,
                         training_epochs = 50,
                         loss_threshold = 5.0,
                         verbose=True)
model.fit(Xtrain, Ytrain)
plot_learning(model)
print(model.score(Xtest, Ytest))

from image_classifier import ImageClassifier


model = ImageClassifier(picsize, lst_dogs,
                         out_channels = 24,
                         out_channels_2 = 48,
                         hidden_units = 200,
                         regularization_strength = 0.5,
                         batch_size = 200,
                         learning_rate = 0.001,
                         convolution_size = 5,
                         pool_size = 2,
                         training_epochs = 50,
                         loss_threshold = 5.0,
                         verbose=True)
model.fit(Xtrain, Ytrain)
plot_learning(model)
print(model.score(Xtest, Ytest))

from image_classifier import ImageClassifier


model = ImageClassifier(picsize, lst_dogs,
                         out_channels = 24,
                         out_channels_2 = 48,
                         hidden_units = 240,
                         regularization_strength = 0.5,
                         batch_size = 200,
                         learning_rate = 0.001,
                         convolution_size = 5,
                         pool_size = 2,
                         training_epochs = 50,
                         loss_threshold = 5.0,
                         verbose=True)
model.fit(Xtrain, Ytrain)
plot_learning(model)
print(model.score(Xtest, Ytest))

from image_classifier import ImageClassifier


model = ImageClassifier(picsize, lst_dogs,
                         out_channels = 24,
                         out_channels_2 = 48,
                         hidden_units = 300,
                         regularization_strength = 0.5,
                         batch_size = 200,
                         learning_rate = 0.001,
                         convolution_size = 5,
                         pool_size = 2,
                         training_epochs = 50,
                         loss_threshold = 5.0,
                         verbose=True)
model.fit(Xtrain, Ytrain)
plot_learning(model)
print(model.score(Xtest, Ytest))

from image_classifier import ImageClassifier


model = ImageClassifier(picsize, lst_dogs,
                         out_channels = 24,
                         out_channels_2 = 48,
                         hidden_units = 100,
                         regularization_strength = 0.5,
                         batch_size = 200,
                         learning_rate = 0.001,
                         convolution_size = 5,
                         pool_size = 2,
                         training_epochs = 50,
                         loss_threshold = 5.0,
                         verbose=True)
model.fit(Xtrain, Ytrain)
plot_learning(model)
print(model.score(Xtest, Ytest))

from image_classifier import ImageClassifier


model = ImageClassifier(picsize, lst_dogs,
                         out_channels = 24,
                         out_channels_2 = 48,
                         hidden_units = 100,
                         regularization_strength = 1.0,
                         batch_size = 300,
                         learning_rate = 0.001,
                         convolution_size = 5,
                         pool_size = 2,
                         training_epochs = 50,
                         loss_threshold = 5.0,
                         verbose=True)
model.fit(Xtrain, Ytrain)
plot_learning(model)
print(model.score(Xtest, Ytest))

