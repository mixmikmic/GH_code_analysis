get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
#%matplotlib nbagg
from Object_detection_features import *
import gym
from gym import wrappers

get_ipython().magic('matplotlib inline')

env = gym.make('Skiing-v0')
# Самая долгая часть. Считается один раз. В конструкторе находятся все классы объектов и фон
odf = ObjectDetectionFeatures(env)

# Генерим картинку
image = env.ale.getScreenGrayscale()
# Создаем новый вектор признаков
new_features = odf.get_distance_features(image)

plt.imshow(image[:,:,0], cmap='Greys')

# на i-й позиции списка cl находится яркость пикселя, характерного для i-го класса
cl = list(odf.all_classes)

# Признаки построены следующем образом:
# Для каждой пары классов находятся 2 ближайших объекта (по центрам масс). 
# Далее для каждой такой пары считаются проекции расстояний на x и на y. 
# В первой позиции стоит проекция на х расстояния между ближайшими объектами из нулевого и первого класса и так далее  
new_features

# на i-й позиции списка cl находится яркость пикселя, характерного для i-го класса
cl = list(odf.all_classes)

# Признаки построены следующем образом:
# Для каждой пары классов находятся 2 ближайших объекта (по центрам масс). 
# Далее для каждой такой пары считаются проекции расстояний на x и на y. 
# В первой позиции стоит проекция на х расстояния между ближайшими объектами из нулевого и первого класса и так далее  
new_features

plt.imshow((image[:, :, 0] == cl[0]) | (image[:, :, 0] == cl[1]))
plt.show()

# Генерим картинку
image = env.ale.getScreenGrayscale()
# Делаем упрощение
new_im = odf.get_simple_image(image[:, :, 0])

plt.imshow(image[:, :, 0])
plt.show()

plt.imshow(new_im)
plt.show()

from skimage.transform import resize
from skimage.color import rgb2gray

plt.imshow(new_im)
plt.show()

import warnings
warnings.filterwarnings('ignore')

max_observations = 20000
observations = []
render = False
count = 0

env_name = 'Skiing-v0'
env = gym.make(env_name)

while True:
    if len(observations) >= max_observations: break
    s = env.reset()
    if count % 10 == 0:
        observation = env.ale.getScreenGrayscale()
        observations.append(odf.get_simple_image(rgb2gray(observation[:, :, 0])))
    count += 1
    done = False

    while not done:
        if render: env.render()
        if len(observations) >= max_observations: break
        a = env.action_space.sample()
        s, r, done, _ = env.step(a)
        if count % 10 == 0:
            observation = env.ale.getScreenGrayscale()
            observations.append(odf.get_simple_image(rgb2gray(observation[:, :, 0])))
            if not len(observations) % 1000:
                print(len(observations))
        a = env.action_space.sample()
        count += 1
        
env.close()

observations = np.array(observations)

np.save('observations_simple_20k.npy', observations)

observations = np.load('observations_simple_20k.npy')

observations.shape

observations = observations.reshape(20000, 1, 60, 60)

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.models import Model
border_mode = 'same'

input_img = Input(shape=observations.shape[1:])

x = Convolution2D(16, 8, 8, activation='relu', border_mode=border_mode)(input_img)
x = MaxPooling2D((2, 2), border_mode=border_mode)(x)
x = Convolution2D(32, 4, 4, activation='relu', border_mode=border_mode)(x)
x = MaxPooling2D((2, 2), border_mode=border_mode)(x)
filters_shape = x.get_shape()
flattened = Flatten()(x)
flat_shape = flattened.get_shape()
encoded = Dense(64, activation='relu')(flattened)

x = Dense(int(flat_shape[1]), activation='relu')(encoded)
x = Reshape(tuple([int(shp) for shp in filters_shape[1:]]))(x)
x = Convolution2D(32, 4, 4, activation='relu', border_mode=border_mode)(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 8, 8, activation='relu', border_mode=border_mode)(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(1, 3, 3, activation='relu', border_mode=border_mode)(x)

decoded = x

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.summary()

import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(float(logs.get('loss')))
    
    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))

history = LossHistory()
autoencoder.fit(observations, observations,
                nb_epoch=40,
                batch_size=32,
                shuffle=True,
                callbacks=[history])

plt.imshow(autoencoder.predict(np.array([observations[25]]))[0, 0], interpolation='Nearest')

plt.imshow(observations[25][0], interpolation='Nearest')

plt.plot(history.losses)
plt.show()

autoencoder.save_weights('./data/Autoencoder_21_01.h5')
autoencoder.to_json()
import json
with open('./data/Autoencoder_21_01.txt', 'w') as outfile:
    json.dump(autoencoder.to_json(), outfile)

encoder.save_weights('./data/Encoder_21_01.h5')
autoencoder.to_json()
import json
with open('./data/Encoder_21_01.txt', 'w') as outfile:
    json.dump(encoder.to_json(), outfile)

sample_features = encoder.predict(observations)
np.savez('./data/sample_features_20k.npz', sample_features)

from keras.models import model_from_json
import json

with open('./data/Autoencoder_21_01.txt', 'r') as model_file:
     model = model_from_json(json.loads(next(model_file)))
        
model.load_weights('./data/Autoencoder_21_01.h5')

max_observations = 1000
test_observations = []
render = False
count = 0

env_name = 'Skiing-v0'
env = gym.make(env_name)

while True:
    if len(test_observations) >= max_observations: break
    s = env.reset()
    if count % 10 == 0:
        observation = env.ale.getScreenGrayscale()
        test_observations.append(odf.get_simple_image(rgb2gray(observation[:, :, 0])))
    count += 1
    done = False

    while not done:
        if render: env.render()
        if len(test_observations) >= max_observations: break
        a = env.action_space.sample()
        s, r, done, _ = env.step(a)
        if count % 10 == 0:
            observation = env.ale.getScreenGrayscale()
            test_observations.append(odf.get_simple_image(rgb2gray(observation[:, :, 0])))
            if not len(test_observations) % 100:
                print(len(test_observations))
        a = env.action_space.sample()
        count += 1
        
env.close()

plt.imshow(test_observations[100], interpolation='Nearest')

test_observations = np.array(test_observations)

test_observations = test_observations.reshape(1000, 1, 60, 60)

predicted = model.predict(test_observations)

np.linalg.norm(predicted - test_observations)**2 / test_observations.size

plt.imshow(test_observations[25][0], interpolation='Nearest')

plt.imshow(model.predict(np.array([test_observations[25]]))[0, 0], interpolation='Nearest')



