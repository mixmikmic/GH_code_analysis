get_ipython().run_cell_magic('bash', '', 'yes|pip2 uninstall captcha\nyes|pip2 uninstall Pillow\npip2 install -U Pillow\npip2 install captcha')

from io import BytesIO
from captcha.image import ImageCaptcha,WheezyCaptcha
from IPython.display import Image

captcha = ImageCaptcha()
data = captcha.generate('1234')
captcha.write('1234', 'out.png')
Image("out.png")

wheezy_captcha = WheezyCaptcha()
captcha.generate('567890')
captcha.write('567890', 'out.png')
Image("out.png")

get_ipython().run_cell_magic('bash', '', 'cat /etc/fonts/fonts.conf')

get_ipython().run_cell_magic('bash', '', 'ls /usr/share/fonts/truetype/*')

import glob

fonts = glob.glob('/usr/share/fonts/truetype/dejavu/*.ttf')

captcha = ImageCaptcha(fonts=fonts)
data = captcha.generate('1234')
captcha.write('1234', 'out.png')
Image("out.png")

wheezy_captcha = WheezyCaptcha(fonts=fonts)
captcha.generate_image('567890')

captcha = ImageCaptcha(fonts=fonts)
captcha.generate_image('a34c')

import numpy as np
import PIL

img = captcha.generate_image('9e3o')
arr = np.asarray(img, dtype="float32")/255.0

data = np.empty((1, 3, 60, 160), dtype="float32")
data[0, :, :, :] = np.rollaxis(arr, 2)

help(img.resize)

img.resize((200, 100))

img.resize((200, 100), PIL.Image.LANCZOS)

from numpy import argmax, array
from sklearn.cross_validation import train_test_split
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Graph
from keras.utils import np_utils
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import math
import random
import IPython.display as display

SAMPLE_SIZE = 1000
SHOW_SAMPLE_SIZE = 5
INVALID_DIGIT = -1
DIGIT_COUNT = 4
DIGIT_FORMAT_STR = "%%0%dd" % DIGIT_COUNT
# print DIGIT_FORMAT_STR
labels = []
images = []
for i in range(0, SAMPLE_SIZE):
    digits = 0
    last_digit = INVALID_DIGIT
    for j in range(0, DIGIT_COUNT):
        digit = last_digit
        while digit == last_digit:
            digit = random.randint(0, 9)
        last_digit = digit
        digits = digits * 10 + digit
    digits_as_str = DIGIT_FORMAT_STR % digits
    labels.append(digits_as_str)
    images.append(captcha.generate_image(digits_as_str))

for index in range(SHOW_SAMPLE_SIZE):
    display.display(labels[index])
    display.display(images[index])

# standard width for the whole captcha image
IMAGE_STD_WIDTH = 200
# standard height for the whole captcha image
IMAGE_STD_HEIGHT = 80
# when spliting an image into digits, how much wider do we want, as an rate
EXTRA_RATE = 0.15
# how much wider do we want, as a width
EXTRA_WIDTH = int(math.floor(IMAGE_STD_WIDTH * EXTRA_RATE))
# the standard digit image width
DIGIT_IMAGE_STD_WIDTH_WITH_EXTRA = IMAGE_STD_WIDTH / DIGIT_COUNT + EXTRA_WIDTH

digit_labels = np.empty(SAMPLE_SIZE * DIGIT_COUNT)
digit_image_data = np.empty((SAMPLE_SIZE * DIGIT_COUNT, 3, IMAGE_STD_HEIGHT, DIGIT_IMAGE_STD_WIDTH_WITH_EXTRA), dtype="float32")

for index in range(0, SHOW_SAMPLE_SIZE):
    img = images[index].resize((IMAGE_STD_WIDTH, IMAGE_STD_HEIGHT), PIL.Image.LANCZOS)
    display.display(img)
    for digit_index in range(0, DIGIT_COUNT):
        #  (left, upper, right, lower)
        left = max(0, IMAGE_STD_WIDTH * (digit_index + 0.0) / DIGIT_COUNT - EXTRA_WIDTH)
        right = min(IMAGE_STD_WIDTH, IMAGE_STD_WIDTH * (digit_index + 1.0) / DIGIT_COUNT + EXTRA_WIDTH)
        crop_box = (left, 0, right, IMAGE_STD_HEIGHT)
        processed_img = img.crop(crop_box)
        processed_img = processed_img.resize((DIGIT_IMAGE_STD_WIDTH_WITH_EXTRA, IMAGE_STD_HEIGHT), PIL.Image.LANCZOS)
        display.display(processed_img)
        img_arr = np.asarray(processed_img, dtype="float32") / 255.0
        digit_image_data[index * DIGIT_COUNT + digit_index, :, :, :] = np.rollaxis(img_arr, 2)

# standard width for the whole captcha image
IMAGE_STD_WIDTH = 200
# standard height for the whole captcha image
IMAGE_STD_HEIGHT = 200

digit_labels = list()

for digit_index in range(0, DIGIT_COUNT):
    digit_labels.append(np.empty(SAMPLE_SIZE, dtype="int8"))
    
digit_image_data = np.empty((SAMPLE_SIZE, 3, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH), dtype="float32")

for index in range(0, SAMPLE_SIZE):
    img = images[index].resize((IMAGE_STD_WIDTH, IMAGE_STD_HEIGHT), PIL.Image.LANCZOS)
    if index < SHOW_SAMPLE_SIZE:
        display.display(img)
    img_arr = np.asarray(img, dtype="float32") / 255.0
    digit_image_data[index, :, :, :] = np.rollaxis(img_arr, 2)
    for digit_index in range(0, DIGIT_COUNT):
        digit_labels[digit_index][index] = labels[index][digit_index]

digit_labels

digit_labels[0][0]

digit_image_data.size == SAMPLE_SIZE * 3 * IMAGE_STD_HEIGHT * IMAGE_STD_WIDTH

digit_image_data[0].shape

# goal is (80,200,3)
np.rollaxis(digit_image_data[0], 0, 3).shape

img = captcha.generate_image('1234')
data = np.asarray(img, dtype="float32") / 255.0
PIL.Image.fromarray(np.uint8(data * 255.0), 'RGB')

display.display(PIL.Image.fromarray(np.uint8(np.rollaxis(digit_image_data[0], 0, 3) * 255.0), 'RGB'))

X, Y_all = digit_image_data, digit_labels[0] 
X_train, X_test, y_train_num, y_test = train_test_split(X, Y_all, test_size=0.1, random_state=0)

print y_train_num

for img in X_train[0:SHOW_SAMPLE_SIZE]:
    display.display(PIL.Image.fromarray(np.uint8(np.rollaxis(img, 0, 3) * 255.0), 'RGB'))

CLASS_COUNT = 10

y_train = np_utils.to_categorical(y_train_num, CLASS_COUNT)
y_train

RGB_COLOR_COUNT = 3

POOL_SIZE = (2, 2)

CONV1_NB_FILTERS = IMAGE_STD_HEIGHT / 2 + 2
CONV2_NB_FILTERS = IMAGE_STD_HEIGHT + 2 * 2

graph = Graph()
# graph.add_input(name='input', input_shape=(3, 40, 40))
graph.add_input(name='input', input_shape=(RGB_COLOR_COUNT, IMAGE_STD_HEIGHT, IMAGE_STD_WIDTH))
# http://stackoverflow.com/questions/36243536/what-is-the-number-of-filter-in-cnn/36243662
graph.add_node(Convolution2D(22, 5, 5, activation='relu'), name='conv1', input='input')
graph.add_node(MaxPooling2D(pool_size=POOL_SIZE), name='pool1', input='conv1')
graph.add_node(Convolution2D(44, 3, 3, activation='relu'), name='conv2', input='pool1')
graph.add_node(MaxPooling2D(pool_size=POOL_SIZE), name='pool2', input='conv2')
graph.add_node(Dropout(0.25), name='drop', input='pool2')
graph.add_node(Flatten(), name='flatten', input='drop')
graph.add_node(Dense(256, activation='relu'), name='ip', input='flatten')
graph.add_node(Dropout(0.5), name='drop_out', input='ip')
graph.add_node(Dense(CLASS_COUNT, activation='softmax'), name='result', input='drop_out')
graph.add_output(name='out', input='result')

graph.compile(
    optimizer='adadelta',
    loss={
        'out': 'categorical_crossentropy',
    }
)

class ValidateAcc(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print '\n————————————————————————————————————'
        graph.load_weights('tmp/weights.%02d.hdf5' % epoch)
        r = graph.predict({'input': X_test}, verbose=0)
        y_predict = array([argmax(i) for i in r['out']])
        length = len(y_predict) * 1.0
        acc = sum(y_predict == y_test) / length
        print 'Single picture test accuracy: %2.2f%%' % (acc * 100)
        print 'Theoretical accuracy: %2.2f%% ~  %2.2f%%' % ((5*acc-4)*100, pow(acc, 5)*100)
        print '————————————————————————————————————'

get_ipython().run_cell_magic('bash', '', 'rm -rf tmp\nrm -rf model\nmkdir tmp\nmkdir model')

check_point = ModelCheckpoint(filepath="tmp/weights.{epoch:02d}.hdf5")
back = ValidateAcc()
print 'Begin train on %d samples... test on %d samples...' % (len(y_train), len(y_test))
graph.fit(
    {'input': X_train, 'out': y_train},
    batch_size=128, nb_epoch=3, callbacks=[check_point, back]
)
print '... saving'
graph.save_weights('model/model_2.hdf5')

check_point = ModelCheckpoint(filepath="tmp/weights.{epoch:02d}.hdf5")
back = ValidateAcc()
print 'Begin train on %d samples... test on %d samples...' % (len(y_train), len(y_test))
graph.load_weights('model/model_2.hdf5')
graph.fit(
    {'input': X_train, 'out': y_train},
    batch_size=128, nb_epoch=5, callbacks=[check_point, back]
)
print '... saving'
graph.save_weights('model/model_2.hdf5', overwrite=True)

check_point = ModelCheckpoint(filepath="tmp/weights.{epoch:02d}.hdf5")
back = ValidateAcc()
print 'Begin train on %d samples... test on %d samples...' % (len(y_train), len(y_test))
graph.load_weights('model/model_2.hdf5')
graph.fit(
    {'input': X_train, 'out': y_train},
    batch_size=128, nb_epoch=10, callbacks=[check_point, back]
)
print '... saving'
graph.save_weights('model/model_2.hdf5', overwrite=True)



