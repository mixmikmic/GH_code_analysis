from plots import plot_recons

import scipy as sp

from plots import plot_vox, plot_dots, plot_compare_recons

import numpy as np
from keras.utils import to_categorical


from data import load_data, load_custom_model

(x_train, y_train), (x_test, y_test), target_names = load_data('./ModelNet40/')
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

def accuracy(y_pred, y_test):
    return np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]

eval_model = load_custom_model('./models/eval_model_net40_acc_08047.hdf5')

manipulate_model = load_custom_model('./models/manipulate_model_net40_acc_08047.hdf5')

from keras.models import Model
get_ipython().run_line_magic('matplotlib', 'inline')


latent_mask = Model(eval_model.input, eval_model.layers[-3].output)
corpus_mask = latent_mask.predict(x_test)
# latent_capsule = Model(eval_model.input, eval_model.layers[-4].output)
# corpus_capsule = latent_capsule.predict(x_test)

def compare(num_1, num_2):
    # plot_dots(x_test[num_1].reshape(30, 30, 30)*.99, dotsize_scale=10, dotsize_offset=0)
    # plot_dots(x_test[num_2].reshape(30, 30, 30)*.99, dotsize_scale=10, dotsize_offset=0)
    plot_vox(x_test[num_1], x_test[num_2])

    print('length', sp.spatial.distance.cosine(corpus_length[num_1], corpus_length[num_2]))
    # print('capsule', sp.spatial.distance.cosine(corpus_capsule[num_1], corpus_capsule[num_2]))
    print('mask', sp.spatial.distance.cosine(corpus_mask[num_1], corpus_mask[num_2]))

num1 = 100
num2 = 101
plot_vox(x_test[num1], x_test[num2])
a = corpus_mask[num1]
b = corpus_mask[num2]
print('Left: {}\t\tRight: {}'.format(target_names[np.argmax(y_test[num1])], target_names[np.argmax(y_test[num2])]))
print('Cosine Similarity via Mask Layer: {}'.format(1 - sp.spatial.distance.cosine(a, b)))

plot_compare_recons(x_test[num1], x_test[num2], y_test[num1], y_test[num2],
                    dim_sub_capsule=16, manipulate_model=manipulate_model,
                    proba_range=[-0.5, 0, 0.5], dotsize_scale=10, dotsize_offset=1,
                    target_names=target_names)



thing = np.zeros((y_test.shape[0], 40))

thing[:, target_names.index('stool')] = 1

thing.shape



x_test[np.argmax(y_test, axis=1) == np.argmax(thing, axis=1)].shape





num1 = 20
num2 = 45
plot_vox(x_test[num1], x_test[num2])
a = corpus_mask[num1]
b = corpus_mask[num2]
print('Left: {}\t\tRight: {}'.format(target_names[np.argmax(y_test[num1])], target_names[np.argmax(y_test[num2])]))
print('Cosine Similarity via Mask Layer: {}'.format(1 - sp.spatial.distance.cosine(a, b)))

plot_compare_recons(x_test[num1], x_test[num2], y_test[num1], y_test[num2],
                    dim_sub_capsule=16, manipulate_model=manipulate_model,
                    proba_range=[-0.5, 0, 0.5], dotsize_scale=10, dotsize_offset=1,
                    target_names=target_names)









plot_recons(x_test[num1], y_test[num1], 16, manipulate_model,
            proba_range=[-0.5, 0, 0.5], dotsize_scale=10,
            dotsize_offset=1)

plot_recons(x_test[num2], y_test[num2], 16, manipulate_model,
            proba_range=[-0.5, 0, 0.5], dotsize_scale=10,
            dotsize_offset=1)







y_test[0]

y_test[100]

y_test[200]

y_test[300]

y_test[400]

import vispy

vispy.test()

for i in [0, 100, 200, 300]:
    for j in [5, 105, 205, 305]:
        compare(i, j)

import vispy.app
vispy.app.use_app('pyqt5')
from vispy.plot import Fig

fig = Fig()

# ax_left = fig[0, 0]
# ax_right = fig[0, 1]
# data = np.random.randn(2, 10)
# ax_left.plot(data)
# ax_right.histogram(data[1])































